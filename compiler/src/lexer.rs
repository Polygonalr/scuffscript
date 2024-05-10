use std::{iter::Peekable, rc::Rc};

#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub source_path: Rc<str>,
    pub line: usize,
    pub col: usize,
}

impl ToString for SourceLocation {
    fn to_string(&self) -> String {
        format!("{}:{}:{}", self.source_path, self.line, self.col)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenData {
    // Words & Keywords
    Identifier(String),
    Function,
    Let,
    Return,
    Print,
    // Types
    IntT,
    DoubleT,
    StringT,

    // Numbers
    Integer(i64),
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
    Colon,
    Comma,
    Equal,
    EqualEqual,
    Ampersand,
    AmpersandAmpersand,
    Bar,
    BarBar,
    Eof,
}

impl From<String> for TokenData {
    fn from(other: String) -> TokenData {
        TokenData::Identifier(other)
    }
}

impl From<i64> for TokenData {
    fn from(other: i64) -> TokenData {
        TokenData::Integer(other)
    }
}

impl From<f64> for TokenData {
    fn from(other: f64) -> TokenData {
        TokenData::Double(other)
    }
}

impl TokenData {
    pub fn is_type(&self) -> bool {
        matches!(
            self,
            TokenData::IntT | TokenData::DoubleT | TokenData::StringT
        )
    }

    pub fn is_term(&self) -> bool {
        matches!(
            self,
            TokenData::Identifier(_) | TokenData::Integer(_) | TokenData::Double(_)
        )
    }
}

#[derive(Debug)]
pub struct Token {
    pub data: TokenData,
    pub loc: SourceLocation,
}

struct Tokenizer<'a> {
    it: Peekable<std::str::Chars<'a>>,
    /// Path to the source file of the code
    source_path: Rc<str>,
    next_char_line: usize,
    next_char_col: usize,
    cache_line: usize,
    cache_col: usize,
}

impl<'a> Tokenizer<'a> {
    fn new(src: &str) -> Tokenizer {
        Tokenizer {
            it: src.chars().peekable(),
            source_path: Rc::from("test"),
            next_char_line: 1,
            next_char_col: 1,
            cache_line: 0,
            cache_col: 0,
        }
    }

    fn next(&mut self) -> Option<char> {
        if let Some('\n') = self.it.peek() {
            self.next_char_line += 1;
            self.next_char_col = 1;
        } else {
            self.next_char_col += 1;
        }
        self.it.next()
    }

    /// Caches the next character's location. To be called at the start of
    /// any tokenization.
    fn cache_loc(&mut self) {
        self.cache_line = self.next_char_line;
        self.cache_col = self.next_char_col;
    }

    fn create_token(&self, token_data: TokenData) -> Token {
        Token {
            data: token_data,
            loc: SourceLocation {
                source_path: self.source_path.clone(),
                line: self.cache_line,
                col: self.cache_col,
            },
        }
    }

    fn consume_while<F: Fn(char) -> bool>(&mut self, pred: F) -> String {
        let mut s = "".to_string();
        while let Some(c) = self.it.peek() {
            if pred(*c) {
                match self.next() {
                    Some(ch) => s.push(ch),
                    None => return s,
                }
            } else {
                break;
            }
        }
        s
    }

    fn tokenize_numeric(&mut self) -> Result<TokenData, String> {
        self.cache_loc();
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
            return Ok(TokenData::from(s.parse::<f64>().unwrap()));
        }
        Ok(TokenData::from(s.parse::<i64>().unwrap()))
    }

    fn tokenize_word(&mut self) -> Result<TokenData, String> {
        self.cache_loc();
        let word = self.consume_while(|x| x.is_alphanumeric());
        match word.as_str() {
            "function" => Ok(TokenData::Function),
            "let" => Ok(TokenData::Let),
            "return" => Ok(TokenData::Return),
            "print" => Ok(TokenData::Print),
            "int" => Ok(TokenData::IntT),
            "double" => Ok(TokenData::DoubleT),
            "string" => Ok(TokenData::StringT),
            "" => Err("Empty string".to_string()),
            _ => Ok(TokenData::from(word.to_string())),
        }
    }

    fn tokenize_string(&mut self) -> Result<TokenData, String> {
        // TODO deal with unexpected new lines between two quotes
        // consume first quote
        self.next();
        let string_data = self.consume_while(|x| x != '"');
        let next_char = self.it.peek();
        if next_char.is_none() {
            return Err("No matching quotation marks for string value".to_string());
        }
        self.next();
        Ok(TokenData::QuotedString(string_data.to_string()))
    }

    fn tokenize_single(&mut self) -> Result<TokenData, String> {
        let c = self.next().unwrap();
        match c {
            '+' => Ok(TokenData::Plus),
            '-' => Ok(TokenData::Minus),
            '*' => Ok(TokenData::Times),
            '/' => Ok(TokenData::Divide),
            '(' => Ok(TokenData::LParen),
            ')' => Ok(TokenData::RParen),
            '[' => Ok(TokenData::LBracket),
            ']' => Ok(TokenData::RBracket),
            '{' => Ok(TokenData::LCurly),
            '}' => Ok(TokenData::RCurly),
            '.' => Ok(TokenData::Dot),
            ';' => Ok(TokenData::Semicolon),
            ':' => Ok(TokenData::Colon),
            ',' => Ok(TokenData::Comma),
            '=' => {
                if let Some(&'=') = self.it.peek() {
                    self.next();
                    Ok(TokenData::EqualEqual)
                } else {
                    Ok(TokenData::Equal)
                }
            }
            '&' => {
                if let Some(&'&') = self.it.peek() {
                    self.next();
                    Ok(TokenData::AmpersandAmpersand)
                } else {
                    Ok(TokenData::Ampersand)
                }
            }
            '|' => {
                if let Some(&'|') = self.it.peek() {
                    self.next();
                    Ok(TokenData::BarBar)
                } else {
                    Ok(TokenData::Bar)
                }
            }
            _ => Err(format!("Unexpected character: {}", c).to_string()),
        }
    }

    fn next_token(&mut self) -> Result<Token, String> {
        self.cache_loc();
        while let Some(ch) = self.it.peek() {
            let token_data = match ch {
                ch if ch.is_whitespace() => {
                    self.next();
                    continue;
                }
                '0'..='9' => self.tokenize_numeric()?,
                'a'..='z' | 'A'..='Z' => self.tokenize_word()?,
                '"' => self.tokenize_string()?,
                _ => self.tokenize_single()?,
            };
            return Ok(self.create_token(token_data));
        }
        self.cache_loc();
        Ok(self.create_token(TokenData::Eof))
    }
}

pub fn tokenize(src: &str) -> Result<Vec<Token>, String> {
    let mut tokenizer = Tokenizer::new(src);
    let mut tokens = Vec::new();
    loop {
        let token = tokenizer.next_token()?;
        let data = token.data.clone();
        tokens.push(token);
        if data == TokenData::Eof {
            break;
        }
    }
    Ok(tokens)
}

fn token_data_map(tokens: Vec<Token>) -> Vec<TokenData> {
    tokens.iter().map(|x| x.data.clone()).collect()
}

#[cfg(test)]
mod tests {

    #[test]
    fn lexer_tests() {
        use crate::lexer::{token_data_map, tokenize, TokenData};
        assert_eq!(
            token_data_map(tokenize("123").unwrap()),
            vec![TokenData::Integer(123), TokenData::Eof]
        );
        assert_eq!(
            token_data_map(tokenize("function main: int {}").unwrap()),
            vec![
                TokenData::Function,
                TokenData::Identifier("main".to_string()),
                TokenData::Colon,
                TokenData::IntT,
                TokenData::LCurly,
                TokenData::RCurly,
                TokenData::Eof
            ]
        );
        assert_eq!(
            token_data_map(
                tokenize("function main: int {\n    print(\"Hello world!\");\n}").unwrap()
            ),
            vec![
                TokenData::Function,
                TokenData::Identifier("main".to_string()),
                TokenData::Colon,
                TokenData::IntT,
                TokenData::LCurly,
                TokenData::Print,
                TokenData::LParen,
                TokenData::QuotedString("Hello world!".to_string()),
                TokenData::RParen,
                TokenData::Semicolon,
                TokenData::RCurly,
                TokenData::Eof,
            ]
        );
    }
}
