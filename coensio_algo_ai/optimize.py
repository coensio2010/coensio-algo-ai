"""GA optimize, DEAP-style GA loop + Rust evaluate_batch.

coensio-algo-ai - powered by coinsio.com
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from coensio_algo_ai import genome as G
from coensio_algo_ai.config import load_engine_cfg, load_raw_cfg
from coensio_algo_ai.fills import EngineConfig, evaluate_batch, register_dataset
from coensio_algo_ai.fitness import composite_fitness, load_fitness_specs
from coensio_algo_ai.genome_fmt import format_genome_str
from coensio_algo_ai.progress import (
    format_progress_line,
    metrics_to_fitness_dict,
    print_completion_banner,
    print_progress_line,
    print_strategy_table,
    top_unique_genome_rows,
)
from coensio_algo_ai.reporting import finalize_ga_report
from coensio_algo_ai.strategy import params_to_genome_vec
from coensio_algo_ai.workers import configure_rayon_workers


@dataclass
class GenomeResult:
    genome: str
    params: dict[str, Any]
    fitness: float
    metrics: dict


@dataclass
class Individual:
    params: dict[str, Any]
    fitness: float = -1e18
    metrics: dict = field(default_factory=dict)
    genome: str = ""
    valid: bool = False


def _resolve_seed(seed: int) -> int:
    """seed<=0 -> fresh random seed each run (fresh exploration each run)."""
    if int(seed) > 0:
        return int(seed)
    return int(secrets.randbelow(2**31 - 1) + 1)


def _tournament_select(
    population: list[Individual], count: int, tournament_size: int, rng
) -> list[Individual]:
    """Tournament selection: for each slot, max of k random picks."""
    out: list[Individual] = []
    n = len(population)
    k = min(tournament_size, n)
    for _ in range(count):
        picks = [population[int(rng.integers(0, n))] for _ in range(k)]
        winner = max(picks, key=lambda ind: ind.fitness)
        out.append(
            Individual(
                params=dict(winner.params),
                fitness=winner.fitness,
                metrics=dict(winner.metrics),
                genome=winner.genome,
                valid=True,
            )
        )
    return out


def run_ga(
    df: pd.DataFrame,
    mod,
    *,
    strategy_id: str,
    datafile: str,
    population: int,
    generations: int,
    seed: int,
    min_trades: int,
    config: EngineConfig,
    session_name: str = "none",
    restart_generations: int | None = None,
    workers: int | None = None,
    report: bool = True,
) -> list[GenomeResult]:
    ecfg = load_engine_cfg()
    raw_cfg = load_raw_cfg()
    fitness_specs = load_fitness_specs(raw_cfg)
    sess = session_name
    restart = (
        restart_generations
        if restart_generations is not None
        else ecfg.restart_generations
    )
    req_workers = ecfg.workers if workers is None else int(workers)
    n_workers = configure_rayon_workers(req_workers, len(df))
    resolved_seed = _resolve_seed(seed)
    rng = np.random.default_rng(resolved_seed)
    spec = mod.PARAMS
    bet_label = f"${config.fixed_bet_size:.0f} {config.bet_mode}"
    data_label = datafile or str(df.attrs.get("datafile", ""))
    cxpb = float(ecfg.cxpb)
    mutpb = float(ecfg.mutpb)
    tourney = int(ecfg.tournament_size)

    dataset_id = register_dataset(df)

    start = df.attrs.get("is_start")
    cut = df.attrs.get("oos_cutoff")
    range_bits = []
    if start:
        range_bits.append(f"is_start={start}")
    if cut:
        range_bits.append(f"oos_cutoff={cut}")
    cut_label = (" | " + " ".join(range_bits)) if range_bits else ""
    print(
        f"GA | {strategy_id} | {data_label} | session={sess} | "
        f"{len(df):,} bars{cut_label} | pop={population} gen={generations} | "
        f"workers={n_workers} (Rust Rayon) | seed={resolved_seed} | coesnio",
        flush=True,
    )
    print(
        f"###GA: strategy={strategy_id} session={sess} pop={population} gen={generations} "
        f"min_trades={min_trades} workers={n_workers} seed={resolved_seed}",
        flush=True,
    )

    best_fitness = -1e18
    best_metrics: dict = {}
    best_genome = ""
    best_params: dict[str, Any] = {}
    db: dict[str, GenomeResult] = {}
    last_batch_ms = 0.0

    def evaluate_population(pop: list[Individual]) -> None:
        nonlocal best_fitness, best_metrics, best_genome, best_params, last_batch_ms
        genomes = [params_to_genome_vec(spec, G.clip_params(spec, ind.params)) for ind in pop]
        t0 = time.perf_counter()
        metrics_list = evaluate_batch(dataset_id, strategy_id, genomes, config=config)
        last_batch_ms = (time.perf_counter() - t0) * 1000.0
        # de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
        for ind, m in zip(pop, metrics_list, strict=True):
            params = G.clip_params(spec, ind.params)
            metrics = metrics_to_fitness_dict(m)
            fit = composite_fitness(
                metrics,
                fitness_specs,
                min_trades,
                initial_capital=config.initial_capital,
            )
            gstr = format_genome_str(
                strategy_id,
                spec,
                params,
                session=sess,
                bet_mode=config.bet_mode,
                fixed_bet_size=config.fixed_bet_size,
                price_bet_frac=config.price_bet_frac,
            )
            ind.params = params
            ind.fitness = fit
            ind.metrics = metrics
            ind.genome = gstr
            ind.valid = True
            prev = db.get(gstr)
            if prev is None or fit > prev.fitness:
                db[gstr] = GenomeResult(genome=gstr, params=params, fitness=fit, metrics=metrics)
            if fit > best_fitness:
                best_fitness = fit
                best_metrics = dict(metrics)
                best_genome = gstr
                best_params = dict(params)

    print(f"###GA: evaluating initial population ({population})...", flush=True)
    pop = [Individual(params=G.random_params(spec, rng)) for _ in range(population)]
    evaluate_population(pop)

    total_generations = 0
    num_generations = generations
    while total_generations < num_generations:
        if total_generations > 0 and total_generations % restart == 0:
            pop = [Individual(params=G.random_params(spec, rng)) for _ in range(population)]
            evaluate_population(pop)

        total_generations += 1
        # Tournament-select full replacement population (no forced elites).
        offspring = _tournament_select(pop, population, tourney, rng)

        for i in range(0, len(offspring) - 1, 2):
            if float(rng.random()) < cxpb:
                c1, c2 = G.crossover(spec, offspring[i].params, offspring[i + 1].params, rng)
                offspring[i] = Individual(params=c1, valid=False)
                offspring[i + 1] = Individual(params=c2, valid=False)

        for i, ind in enumerate(offspring):
            if float(rng.random()) < mutpb:
                offspring[i] = Individual(
                    params=G.mutate_params(spec, ind.params, rng),
                    valid=False,
                )

        invalid = [ind for ind in offspring if not ind.valid]
        if invalid:
            evaluate_population(invalid)
        pop = offspring

        ms_per_str = last_batch_ms / population if population else 0.0
        cycle = (total_generations - 1) // restart + 1
        cycle_gen = (total_generations - 1) % restart + 1
        prefix = (
            f"C:{cycle:02d}G:{cycle_gen:03d}"
            f"T:{total_generations:05d}/{num_generations:05d} "
        )
        line = format_progress_line(
            best_metrics,
            best_fit=best_fitness,
            ms_per_str=ms_per_str,
            session=sess,
            bet_label=bet_label,
            prefix=prefix,
        )
        print_progress_line(line)

    print("", flush=True)
    print_completion_banner(best_genome, best_fitness)

    ranked = sorted(db.values(), key=lambda x: x.fitness, reverse=True)
    table_rows = top_unique_genome_rows(
        ranked,
        ecfg.table_top_n,
        min_trades=min_trades,
        datafile=data_label,
    )
    print_strategy_table(
        table_rows,
        title=f"Top {ecfg.table_top_n} unique genomes",
        min_trades=min_trades,
    )
    print(f"\nBest genome: {best_genome}")

    if report:
        finalize_ga_report(
            strategy_id=strategy_id,
            genome_str=best_genome,
            params=best_params,
            df=df,
            datafile=data_label,
            session=sess,
            config=config,
            raw_cfg=raw_cfg,
        )
    return ranked
