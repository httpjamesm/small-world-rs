use half::f16;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
enum VectorStorage {
    F16(Vec<f16>),
    F32(Vec<f32>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vector {
    storage: VectorStorage,
}

impl Vector {
    pub fn new_f32(values: &[f32]) -> Self {
        Self {
            storage: VectorStorage::F32(values.to_vec()),
        }
    }

    pub fn new_f16(values: &[f32]) -> Self {
        Self {
            storage: VectorStorage::F16(values.iter().map(|&x| f16::from_f32(x)).collect()),
        }
    }

    pub fn len(&self) -> usize {
        match &self.storage {
            VectorStorage::F16(v) => v.len(),
            VectorStorage::F32(v) => v.len(),
        }
    }

    // Get value at index as f32
    pub fn get(&self, index: usize) -> Option<f32> {
        match &self.storage {
            VectorStorage::F16(v) => v.get(index).map(|x| x.to_f32()),
            VectorStorage::F32(v) => v.get(index).copied(),
        }
    }
}

pub enum VectorIter<'a> {
    F16(std::iter::Map<std::slice::Iter<'a, f16>, fn(&f16) -> f32>),
    F32(std::iter::Copied<std::slice::Iter<'a, f32>>),
}

impl<'a> Iterator for VectorIter<'a> {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            VectorIter::F16(iter) => iter.next(),
            VectorIter::F32(iter) => iter.next(),
        }
    }
}

impl Vector {
    pub fn iter(&self) -> VectorIter {
        match &self.storage {
            VectorStorage::F16(v) => VectorIter::F16(v.iter().map(|x| x.to_f32())),
            VectorStorage::F32(v) => VectorIter::F32(v.iter().copied()),
        }
    }
}
