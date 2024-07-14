use std::rc::Rc;

pub struct MlirIdGen {
    idx: usize,
}

impl MlirIdGen {
    pub fn new() -> Self {
        Self { idx: 0 }
    }

    pub fn gen_id(&mut self, id: &str) -> Rc<str> {
        let idx = self.idx;
        self.idx += 1;
        Rc::from(format!("{}{}", id, idx))
    }
}
