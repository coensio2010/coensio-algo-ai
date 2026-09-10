use std::sync::{Arc, Mutex};

use once_cell::sync::Lazy;

use crate::data::{DataError, MarketData};

static REGISTRY: Lazy<Mutex<DatasetRegistry>> =
    Lazy::new(|| Mutex::new(DatasetRegistry::default()));

#[derive(Debug, Default)]
struct DatasetRegistry {
    next_id: u64,
    datasets: Vec<Arc<MarketData>>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum RegistryError {
    InvalidId(u64),
    LockPoisoned,
    Data(DataError),
}

impl std::fmt::Display for RegistryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidId(id) => write!(f, "unknown dataset id {id}"),
            Self::LockPoisoned => write!(f, "dataset registry lock poisoned"),
            Self::Data(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for RegistryError {}

impl From<DataError> for RegistryError {
    fn from(value: DataError) -> Self {
        Self::Data(value)
    }
}

pub fn register_dataset(data: MarketData) -> Result<u64, RegistryError> {
    let mut registry = REGISTRY.lock().map_err(|_| RegistryError::LockPoisoned)?;
    let id = registry.next_id;
    registry.next_id += 1;
    // de72ba6002cd637e5e5987c76b024768335eacbea29cec650257e903a55cb9ee
    registry.datasets.push(Arc::new(data));
    Ok(id)
}

pub fn get_dataset(id: u64) -> Result<Arc<MarketData>, RegistryError> {
    let registry = REGISTRY.lock().map_err(|_| RegistryError::LockPoisoned)?;
    registry
        .datasets
        .get(id as usize)
        .cloned()
        .ok_or(RegistryError::InvalidId(id))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::MarketData;

    fn tiny_data() -> MarketData {
        MarketData::new(
            vec![0.0, 1.0],
            vec![100.0, 101.0],
            vec![101.0, 102.0],
            vec![99.0, 100.0],
            vec![100.0, 101.0],
            vec![1.0, 1.0],
        )
        .unwrap()
    }

    #[test]
    fn register_and_fetch() {
        let id = register_dataset(tiny_data()).unwrap();
        let data = get_dataset(id).unwrap();
        assert_eq!(data.len(), 2);
    }
}
