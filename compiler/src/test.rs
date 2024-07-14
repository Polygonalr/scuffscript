use super::*;

#[test]
fn simple1() {
    let input = fs::read_to_string("./tests/simple1/in.scuff")
        .expect("Failed to read tests/simple1/in.scuff");
    let ll_output =
        fs::read_to_string("./tests/simple1/out.ll").expect("Failed to read tests/simple1/out.ll");
    let s_output =
        fs::read_to_string("./tests/simple1/out.s").expect("Failed to read tests/simple1/out.s");
    assert_eq!(
        compile(input.clone(), None, true, false).unwrap(),
        ll_output
    );
    assert_eq!(compile(input, None, false, true).unwrap(), s_output);
}
