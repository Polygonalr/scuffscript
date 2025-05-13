use super::*;

fn test_boilerplate(test_dir: String) {
    let input = fs::read_to_string(format!("./tests/{}/in.scuff", test_dir))
        .expect(&format!("Failed to read tests/{}/in.scuff", test_dir));
    let ll_output =
        fs::read_to_string(format!("./tests/{}/out.ll", test_dir)).expect(&format!("Failed to read tests/{}/out.ll", test_dir));
    let s_output =
        fs::read_to_string(format!("./tests/{}/out.s", test_dir)).expect(&format!("Failed to read tests/{}/out.s", test_dir));
    assert_eq!(
        compile(input.clone(), None, true, false).unwrap(),
        ll_output
    );
    assert_eq!(compile(input, None, false, true).unwrap(), s_output);
}

#[test]
fn simple() {
    test_boilerplate("simple".to_string());
}

#[test]
fn func_call() {
    test_boilerplate("func_call".to_string());
}

#[test]
fn if_else_1() {
    test_boilerplate("if_else_1".to_string());
}

#[test]
fn if_else_2() {
    test_boilerplate("if_else_2".to_string());
}
