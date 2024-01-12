use std::iter::Peekable;

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Words & Keywords
    Identifier(String),
    Function,
    Let,
    Return,
    Print,

    // Numbers
    Integer(i32),
    Double(f64),

    // Strings
    QuotedString(String),

    // Symbols
    Plus,
    Minus,
    Times,
    Divide,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LCurly,
    RCurly,
    Dot,
    Semicolon,
    Eof,
}

impl From<String> for Token {
    fn from(other: String) -> Token {
        Token::Identifier(other)
    }
}

impl From<i32> for Token {
    fn from(other: i32) -> Token {
        Token::Integer(other)
    }
}

impl From<f64> for Token {
    fn from(other: f64) -> Token {
        Token::Double(other)
    }
}

struct Tokenizer<'a> {
    it: Peekable<std::str::Chars<'a>>,
}

impl<'a> Tokenizer<'a> {
    fn new(src: &str) -> Tokenizer {
        Tokenizer {
            it: src.chars().peekable(),
        }
    }

    fn consume_while<F: Fn(char) -> bool>(&mut self, pred: F) -> String {
        let mut s = "".to_string();
        while let Some(c) = self.it.peek() {
            if pred(*c) {
                match self.it.next() {
                    Some(ch) => s.push(ch),
                    None => return s,
                }
            } else {
                break;
            }
        }
        s
    }

    fn tokenize_numeric(&mut self) -> Result<Token, String> {
        // TODO ensure that character following the numeral is not alpha
        let numerals = self.consume_while(|x| x.is_numeric() || x == '.');
        let mut has_period = false;
        let mut s = "".to_string();
        for n in numerals.chars() {
            if n == '.' {
                if has_period {
                    return Err("Error: Invalid numeral".to_string());
                } else {
                    has_period = true;
                    s.push(n);
                }
            } else {
                s.push(n);
            }
        }
        if has_period {
            return Ok(Token::from(s.parse::<f64>().unwrap()));
        }
        Ok(Token::from(s.parse::<i32>().unwrap()))
    }

    fn tokenize_word(&mut self) -> Result<Token, String> {
        let word = self.consume_while(|x| x.is_alphanumeric());
        match word.as_str() {
            "function" => Ok(Token::Function),
            "let" => Ok(Token::Let),
            "return" => Ok(Token::Return),
            "print" => Ok(Token::Print),
            "" => Err("Empty string".to_string()),
            _ => Ok(Token::from(word.to_string())),
        }
    }

    fn tokenize_string(&mut self) -> Result<Token, String> {
        // consume first quote
        self.it.next();
        let string_data = self.consume_while(|x| x != '"');
        let next_char = self.it.peek();
        if next_char.is_none() {
            return Err("No matching quotation marks for string value".to_string());
        }
        self.it.next();
        Ok(Token::QuotedString(string_data.to_string()))
    }

    fn tokenize_single(&mut self) -> Result<Token, String> {
        let c = self.it.next().unwrap();
        match c {
            '+' => Ok(Token::Plus),
            '-' => Ok(Token::Minus),
            '*' => Ok(Token::Times),
            '/' => Ok(Token::Divide),
            '(' => Ok(Token::LParen),
            ')' => Ok(Token::RParen),
            '[' => Ok(Token::LBracket),
            ']' => Ok(Token::RBracket),
            '{' => Ok(Token::LCurly),
            '}' => Ok(Token::RCurly),
            '.' => Ok(Token::Dot),
            ';' => Ok(Token::Semicolon),
            _ => Err(format!("Unexpected character: {}", c).to_string()),
        }
    }

    fn next_token(&mut self) -> Result<Option<Token>, String> {
        while let Some(ch) = self.it.peek() {
            let tkn = match ch {
                ch if ch.is_whitespace() => {
                    self.it.next();
                    continue;
                }
                '0'..='9' => self.tokenize_numeric().unwrap(),
                'a'..='z' | 'A'..='Z' => self.tokenize_word().unwrap(),
                '"' => self.tokenize_string().unwrap(),
                _ => self.tokenize_single().unwrap(),
            };
            return Ok(Some(tkn));
        }
        Ok(None)
    }
}

pub fn tokenize(src: &str) -> Result<Vec<Token>, String> {
    let mut tokenizer = Tokenizer::new(src);
    let mut tokens = Vec::new();
    loop {
        let token = tokenizer.next_token()?;
        if let Some(t) = token {
            tokens.push(t);
        } else {
            break;
        }
    }
    Ok(tokens)
}

#[cfg(test)]
mod tests {
    #[test]
    fn lexer_tests() {
        use crate::lexer::{tokenize, Token};
        assert_eq!(tokenize("123"), Ok(vec![Token::Integer(123)]));
        assert_eq!(
            tokenize("function main {}"),
            Ok(vec![
                Token::Function,
                Token::Identifier("main".to_string()),
                Token::LCurly,
                Token::RCurly
            ])
        );
        assert_eq!(
            tokenize("function main {\n    print(\"Hello world!\");\n}"),
            Ok(vec![
                Token::Function,
                Token::Identifier("main".to_string()),
                Token::LCurly,
                Token::Print,
                Token::LParen,
                Token::QuotedString("Hello world!".to_string()),
                Token::RParen,
                Token::Semicolon,
                Token::RCurly
            ])
        );
    }
}
