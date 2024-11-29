use std::array::TryFromSliceError;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DatabaseError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),

    #[error("Invalid SQLite file format")]
    InvalidFormat,

    #[error("Unsupported SQLite version")]
    UnsupportedVersion,

    #[error("Invalid page type: {0}")]
    InvalidPageType(u8),

    #[error("Invalid B-tree structure")]
    InvalidBtreeStructure,

    #[error("Invalid varint")]
    InvalidVarint,

    #[error("Invalid record format")]
    InvalidRecordFormat,

    #[error("SQL parse error: {0}")]
    SqlParseError(String),

    #[error("Table not found: {0}")]
    TableNotFound(String),

    #[error("Invalid column reference: {0}")]
    InvalidColumn(String),

    #[error("Array conversion error")]
    ArrayConversionError(#[from] TryFromSliceError),
}

/// Parse a SQLite varint from a byte slice
/// Returns the parsed value and the number of bytes consumed
fn parse_varint(data: &[u8]) -> Result<(u64, usize), DatabaseError> {
    if data.is_empty() {
        return Err(DatabaseError::InvalidVarint);
    }

    let mut result: u64 = 0;
    let mut bytes_read = 0;

    for (i, &byte) in data.iter().take(9).enumerate() {
        bytes_read += 1;

        if i == 8 {
            // Last byte, use all 8 bits
            result = (result << 8) | (byte as u64);
            break;
        } else {
            // Use 7 bits, check continuation bit
            result = (result << 7) | ((byte & 0x7F) as u64);
            if byte & 0x80 == 0 {
                break;
            }
        }
    }

    Ok((result, bytes_read))
}

#[derive(Debug, Clone)]
pub enum SqliteValue {
    Null,
    Integer(i64),
    Float(f64),
    Text(String),
    Blob(Vec<u8>),
}

#[derive(Debug)]
struct Record {
    // header_size: usize,
    // serial_types: Vec<u64>,
    values: Vec<SqliteValue>,
}

impl Record {
    fn parse(data: &[u8]) -> Result<Self, DatabaseError> {
        // Parse record header size
        let (header_size, mut offset) = parse_varint(data)?;
        let header_size = header_size as usize;

        // Parse serial types from header
        let mut serial_types = Vec::new();
        while offset < header_size {
            let (serial_type, bytes_read) = parse_varint(&data[offset..])?;
            serial_types.push(serial_type);
            offset += bytes_read;
        }

        // Parse values according to their serial types
        let mut values = Vec::new();
        for &serial_type in &serial_types {
            if offset >= data.len() {
                return Err(DatabaseError::InvalidRecordFormat);
            }

            let value = match serial_type {
                0 => SqliteValue::Null,
                1 => {
                    if offset + 1 > data.len() {
                        return Err(DatabaseError::InvalidRecordFormat);
                    }
                    let mut bytes = [0u8; 8];
                    bytes[7] = data[offset];
                    let value = i64::from_be_bytes(bytes);
                    offset += 1;
                    SqliteValue::Integer(value)
                }
                2 => {
                    if offset + 2 > data.len() {
                        return Err(DatabaseError::InvalidRecordFormat);
                    }
                    let mut bytes = [0u8; 8];
                    bytes[6..8].copy_from_slice(&data[offset..offset + 2]);
                    let value = i64::from_be_bytes(bytes);
                    offset += 2;
                    SqliteValue::Integer(value)
                }
                3 => {
                    if offset + 3 > data.len() {
                        return Err(DatabaseError::InvalidRecordFormat);
                    }
                    let mut bytes = [0u8; 8];
                    bytes[5..8].copy_from_slice(&data[offset..offset + 3]);
                    let value = i64::from_be_bytes(bytes);
                    offset += 3;
                    SqliteValue::Integer(value)
                }
                4 => {
                    if offset + 4 > data.len() {
                        return Err(DatabaseError::InvalidRecordFormat);
                    }
                    let mut bytes = [0u8; 8];
                    bytes[4..8].copy_from_slice(&data[offset..offset + 4]);
                    let value = i64::from_be_bytes(bytes);
                    offset += 4;
                    SqliteValue::Integer(value)
                }
                5 => {
                    if offset + 6 > data.len() {
                        return Err(DatabaseError::InvalidRecordFormat);
                    }
                    let mut bytes = [0u8; 8];
                    bytes[2..8].copy_from_slice(&data[offset..offset + 6]);
                    let value = i64::from_be_bytes(bytes);
                    offset += 6;
                    SqliteValue::Integer(value)
                }
                6 => {
                    if offset + 8 > data.len() {
                        return Err(DatabaseError::InvalidRecordFormat);
                    }
                    let value = i64::from_be_bytes(data[offset..offset + 8].try_into()?);
                    offset += 8;
                    SqliteValue::Integer(value)
                }
                7 => {
                    if offset + 8 > data.len() {
                        return Err(DatabaseError::InvalidRecordFormat);
                    }
                    let value = f64::from_be_bytes(data[offset..offset + 8].try_into()?);
                    offset += 8;
                    SqliteValue::Float(value)
                }
                8 => SqliteValue::Integer(0),
                9 => SqliteValue::Integer(1),
                n if n >= 13 && n % 2 == 1 => {
                    // Text
                    let len = ((n - 13) / 2) as usize;
                    if offset + len > data.len() {
                        return Err(DatabaseError::InvalidRecordFormat);
                    }
                    let text = String::from_utf8_lossy(&data[offset..offset + len]).to_string();
                    offset += len;
                    SqliteValue::Text(text)
                }
                n if n >= 12 && n % 2 == 0 => {
                    // BLOB
                    let len = ((n - 12) / 2) as usize;
                    if offset + len > data.len() {
                        return Err(DatabaseError::InvalidRecordFormat);
                    }
                    let blob = data[offset..offset + len].to_vec();
                    offset += len;
                    SqliteValue::Blob(blob)
                }
                _ => return Err(DatabaseError::InvalidRecordFormat),
            };
            values.push(value);
        }

        Ok(Record { values })
    }
}

#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub column_type: String,
    pub not_null: bool,
    pub primary_key: bool,
    pub auto_increment: bool,
}

#[derive(Debug)]
struct Table {
    // pub name: String,
    // pub columns: Vec<Column>,
    root_page: u32,
}

// impl Table {
//     fn parse_create_sql(sql: &str) -> Result<Vec<Column>, DatabaseError> {
//         let mut columns = Vec::new();

//         // Find the column definitions between parentheses
//         if let Some(start) = sql.find('(') {
//             if let Some(end) = sql.rfind(')') {
//                 let columns_sql = &sql[start + 1..end];

//                 // Split by commas, but not inside parentheses
//                 let mut depth = 0;
//                 let mut current = String::new();

//                 for c in columns_sql.chars() {
//                     match c {
//                         '(' => {
//                             depth += 1;
//                             current.push(c);
//                         }
//                         ')' => {
//                             depth -= 1;
//                             current.push(c);
//                         }
//                         ',' if depth == 0 => {
//                             if !current.trim().is_empty() {
//                                 if let Some(col) = parse_column_def(&current) {
//                                     columns.push(col);
//                                 }
//                                 current.clear();
//                             }
//                         }
//                         _ => current.push(c),
//                     }
//                 }

//                 if !current.trim().is_empty() {
//                     if let Some(col) = parse_column_def(&current) {
//                         columns.push(col);
//                     }
//                 }
//             }
//         }

//         Ok(columns)
//     }
// }

// fn parse_column_def(def: &str) -> Option<Column> {
//     let parts: Vec<&str> = def.trim().split_whitespace().collect();
//     if parts.len() < 2 {
//         return None;
//     }

//     let name = parts[0]
//         .trim()
//         .trim_matches('"')
//         .trim_matches('`')
//         .to_string();
//     let type_str = parts[1].trim().to_uppercase();

//     let mut column = Column {
//         name,
//         column_type: type_str,
//         not_null: false,
//         primary_key: false,
//         auto_increment: false,
//     };

//     // Parse constraints
//     for i in 2..parts.len() {
//         match parts[i].to_uppercase().as_str() {
//             "NOT" if i + 1 < parts.len() && parts[i + 1].to_uppercase() == "NULL" => {
//                 column.not_null = true;
//             }
//             "PRIMARY" if i + 1 < parts.len() && parts[i + 1].to_uppercase() == "KEY" => {
//                 column.primary_key = true;
//             }
//             "AUTOINCREMENT" => {
//                 column.auto_increment = true;
//             }
//             _ => {}
//         }
//     }

//     Some(column)
// }

#[derive(Debug)]
struct BTreeCell {
    left_child_page: Option<u32>, // Only for interior pages
    // row_id: u64,                  // For table b-trees
    payload: Vec<u8>,
}

#[derive(Debug)]
struct BTreePage {
    page_type: PageType,
    // first_freeblock: u16,
    // cell_count: u16,
    // cell_content_start: u16,
    // fragmented_free_bytes: u8,
    cell_pointers: Vec<u16>,
}

impl BTreePage {
    fn parse(data: &[u8]) -> Result<Self, DatabaseError> {
        let page_type = PageType::try_from(data[0])?;
        // let first_freeblock = u16::from_be_bytes([data[1], data[2]]);
        let cell_count = u16::from_be_bytes([data[3], data[4]]);
        // let cell_content_start = u16::from_be_bytes([data[5], data[6]]);
        // let fragmented_free_bytes = data[7];

        let mut cell_pointers = Vec::with_capacity(cell_count as usize);
        for i in 0..cell_count {
            let offset = 8 + i as usize * 2;
            let pointer = u16::from_be_bytes([data[offset], data[offset + 1]]);
            cell_pointers.push(pointer);
        }

        Ok(BTreePage {
            page_type,
            // first_freeblock,
            // cell_count,
            // cell_content_start,
            // fragmented_free_bytes,
            cell_pointers,
        })
    }

    fn get_cell(&self, data: &[u8], cell_pointer: u16) -> Result<BTreeCell, DatabaseError> {
        let offset = cell_pointer as usize;
        let mut current_offset = offset;

        // Read cell content based on page type
        match self.page_type {
            PageType::LeafTable => {
                // Format: payload_size(varint), row_id(varint), payload
                let (payload_size, size_len) = parse_varint(&data[current_offset..])?;
                current_offset += size_len;

                let (_row_id, row_id_len) = parse_varint(&data[current_offset..])?;
                current_offset += row_id_len;

                let payload = data[current_offset..current_offset + payload_size as usize].to_vec();

                Ok(BTreeCell {
                    left_child_page: None,
                    // row_id,
                    payload,
                })
            }
            PageType::InteriorTable => {
                // Format: left_child_page(4), row_id(varint)
                let left_child = u32::from_be_bytes(
                    data[current_offset..current_offset + 4].try_into().unwrap(),
                );
                // current_offset += 4;

                // let (row_id, _) = parse_varint(&data[current_offset..])?;

                Ok(BTreeCell {
                    left_child_page: Some(left_child),
                    // row_id,
                    payload: Vec::new(),
                })
            }
            _ => {
                println!("Invalid page type: {:?}", self.page_type);
                Err(DatabaseError::InvalidPageType(self.page_type.clone() as u8))
            }
        }
    }

    fn get_cell_content(&self, data: &[u8], cell_pointer: u16) -> Result<Record, DatabaseError> {
        match self.page_type {
            PageType::LeafTable => {
                // For leaf table pages, cell format is:
                // * Payload length (varint)
                // * Row ID (varint)
                // * Payload (Record)
                let offset = cell_pointer as usize;
                let (_payload_len, bytes_read) = parse_varint(&data[offset..])?;
                let (_row_id, row_id_size) = parse_varint(&data[offset + bytes_read..])?;
                let content_offset = offset + bytes_read + row_id_size;
                Record::parse(&data[content_offset..])
            }
            _ => {
                println!("Invalid page type: {:?}", self.page_type);
                Err(DatabaseError::InvalidPageType(self.page_type.clone() as u8))
                // Handle other page types as needed
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Operator {
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
    GreaterEquals,
    LessEquals,
    Like,
}

#[derive(Debug, Clone)]
pub struct WhereClause {
    pub column: String,
    pub operator: Operator,
    pub value: SqliteValue,
}

#[derive(Debug, Clone)]
#[repr(u8)]
enum PageType {
    InteriorIndex = 2,
    InteriorTable = 5,
    LeafIndex = 10,
    LeafTable = 13,
}

impl TryFrom<u8> for PageType {
    type Error = DatabaseError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            2 => Ok(PageType::InteriorIndex),
            5 => Ok(PageType::InteriorTable),
            10 => Ok(PageType::LeafIndex),
            13 => Ok(PageType::LeafTable),
            _ => Err(DatabaseError::InvalidPageType(value)),
        }
    }
}

#[derive(Debug)]
struct DatabaseHeader {
    // /// Magic header string: "SQLite format 3\0"
    // magic: [u8; 16],
    /// Page size in bytes
    page_size: u16,
    // /// File format write version
    // write_version: u8,
    // /// File format read version
    // read_version: u8,
    // /// Bytes of unused space at end of each page
    // reserved_space: u8,
    // /// Maximum embedded payload fraction
    // max_payload_fraction: u8,
    // /// Minimum embedded payload fraction
    // min_payload_fraction: u8,
    // /// Leaf payload fraction
    // leaf_payload_fraction: u8,
    // /// File change counter
    // file_change_counter: u32,
    // /// Size of database file in pages
    // database_size: u32,
    // /// Page number of first freelist trunk page
    // first_freelist_trunk: u32,
    // /// Total number of freelist pages
    // total_freelist_pages: u32,
    // /// Schema format number
    // schema_cookie: u32,
    // /// Schema format number
    // schema_format: u32,
    // /// Default page cache size
    // default_cache_size: u32,
    // /// Page number of largest root b-tree
    // largest_root_btree: u32,
    // /// Database text encoding (1: utf8, 2: utf16le, 3: utf16be)
    // text_encoding: u32,
    // /// User version
    // user_version: u32,
    // /// Incremental vacuum mode
    // incremental_vacuum: u32,
    // /// Application ID
    // application_id: u32,
    // /// Reserved for expansion
    // reserved: [u8; 20],
    // /// Version-valid-for number
    // version_valid_for: u32,
    // /// SQLite version number
    // sqlite_version: u32,
}

pub struct Database {
    file: File,
    header: DatabaseHeader,
}

impl Database {
    /// Opens a SQLite database file at the specified path
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, DatabaseError> {
        let mut file = File::open(path)?;
        let header = Self::read_header(&mut file)?;
        println!("header: {:?}", header);

        Ok(Database { file, header })
    }

    /// Read and parse the SQLite database header
    fn read_header(file: &mut File) -> Result<DatabaseHeader, DatabaseError> {
        let mut header_bytes = [0u8; 100];
        file.seek(SeekFrom::Start(0))?;
        file.read_exact(&mut header_bytes)?;

        // Check magic string
        let magic = &header_bytes[0..16];
        if magic != b"SQLite format 3\0" {
            return Err(DatabaseError::InvalidFormat);
        }

        // Parse header fields
        let mut magic_arr = [0u8; 16];
        magic_arr.copy_from_slice(magic);

        let page_size = u16::from_be_bytes([header_bytes[16], header_bytes[17]]);

        let header = DatabaseHeader {
            // magic: magic_arr,
            page_size,
            // write_version: header_bytes[18],
            // read_version: header_bytes[19],
            // reserved_space: header_bytes[20],
            // max_payload_fraction: header_bytes[21],
            // min_payload_fraction: header_bytes[22],
            // leaf_payload_fraction: header_bytes[23],
            // file_change_counter: u32::from_be_bytes([
            //     header_bytes[24],
            //     header_bytes[25],
            //     header_bytes[26],
            //     header_bytes[27],
            // ]),
            // database_size: u32::from_be_bytes([
            //     header_bytes[28],
            //     header_bytes[29],
            //     header_bytes[30],
            //     header_bytes[31],
            // ]),
            // first_freelist_trunk: u32::from_be_bytes([
            //     header_bytes[32],
            //     header_bytes[33],
            //     header_bytes[34],
            //     header_bytes[35],
            // ]),
            // total_freelist_pages: u32::from_be_bytes([
            //     header_bytes[36],
            //     header_bytes[37],
            //     header_bytes[38],
            //     header_bytes[39],
            // ]),
            // schema_cookie: u32::from_be_bytes([
            //     header_bytes[40],
            //     header_bytes[41],
            //     header_bytes[42],
            //     header_bytes[43],
            // ]),
            // schema_format: u32::from_be_bytes([
            //     header_bytes[44],
            //     header_bytes[45],
            //     header_bytes[46],
            //     header_bytes[47],
            // ]),
            // default_cache_size: u32::from_be_bytes([
            //     header_bytes[48],
            //     header_bytes[49],
            //     header_bytes[50],
            //     header_bytes[51],
            // ]),
            // largest_root_btree: u32::from_be_bytes([
            //     header_bytes[52],
            //     header_bytes[53],
            //     header_bytes[54],
            //     header_bytes[55],
            // ]),
            // text_encoding: u32::from_be_bytes([
            //     header_bytes[56],
            //     header_bytes[57],
            //     header_bytes[58],
            //     header_bytes[59],
            // ]),
            // user_version: u32::from_be_bytes([
            //     header_bytes[60],
            //     header_bytes[61],
            //     header_bytes[62],
            //     header_bytes[63],
            // ]),
            // incremental_vacuum: u32::from_be_bytes([
            //     header_bytes[64],
            //     header_bytes[65],
            //     header_bytes[66],
            //     header_bytes[67],
            // ]),
            // application_id: u32::from_be_bytes([
            //     header_bytes[68],
            //     header_bytes[69],
            //     header_bytes[70],
            //     header_bytes[71],
            // ]),
            // reserved: {
            //     let mut reserved = [0u8; 20];
            //     reserved.copy_from_slice(&header_bytes[72..92]);
            //     reserved
            // },
            // version_valid_for: u32::from_be_bytes([
            //     header_bytes[92],
            //     header_bytes[93],
            //     header_bytes[94],
            //     header_bytes[95],
            // ]),
            // sqlite_version: u32::from_be_bytes([
            //     header_bytes[96],
            //     header_bytes[97],
            //     header_bytes[98],
            //     header_bytes[99],
            // ]),
        };

        Ok(header)
    }

    /// Get database header information
    // pub fn get_header_info(&self) -> &DatabaseHeader {
    //     &self.header
    // }

    /// Read a page from the database
    pub fn read_page(&mut self, page_number: u32) -> Result<Vec<u8>, DatabaseError> {
        let offset = (page_number as u64 - 1) * self.header.page_size as u64;
        let mut page_data = vec![0u8; self.header.page_size as usize];

        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut page_data)?;

        Ok(page_data)
    }

    /// Parse a page as a B-tree page
    fn parse_btree_page(&mut self, page_number: u32) -> Result<BTreePage, DatabaseError> {
        let page_data = self.read_page(page_number)?;

        // The first page contains the database header (100 bytes)
        let data = if page_number == 1 {
            println!("First byte of page after header: {}", page_data[100]);
            &page_data[100..]
        } else {
            &page_data
        };

        BTreePage::parse(data)
    }

    /// Get all tables in the database
    pub fn list_tables(&mut self) -> Result<Vec<String>, DatabaseError> {
        // SQLite stores schema information in a special table named "sqlite_master"
        // It's always stored in the first page after the header
        let schema_page = self.parse_btree_page(1)?;
        let page_data = self.read_page(1)?;

        let mut table_names = Vec::new();

        // Process each cell in the schema page
        for &cell_pointer in &schema_page.cell_pointers {
            if let Ok(record) = schema_page.get_cell_content(&page_data, cell_pointer) {
                // Schema table columns: type, name, tbl_name, rootpage, sql
                if let SqliteValue::Text(type_name) = &record.values[0] {
                    if type_name == "table" {
                        if let SqliteValue::Text(table_name) = &record.values[2] {
                            if !table_name.starts_with("sqlite_") {
                                table_names.push(table_name.clone());
                            }
                        }
                    }
                }
            }
        }

        Ok(table_names)
    }

    /// Extract table name from a simple SELECT query
    fn extract_table_name(sql: &str) -> Result<String, DatabaseError> {
        let sql = sql.trim().to_uppercase();
        if !sql.starts_with("SELECT") {
            return Err(DatabaseError::SqlParseError("Not a SELECT query".to_string()));
        }

        // Find FROM clause
        if let Some(from_idx) = sql.find("FROM") {
            let after_from = &sql[from_idx + 4..];
            // Get the first word after FROM (the table name)
            let table_name = after_from
                .split_whitespace()
                .next()
                .ok_or_else(|| DatabaseError::SqlParseError("No table name found".to_string()))?
                .to_string();
            
            Ok(table_name)
        } else {
            Err(DatabaseError::SqlParseError("No FROM clause found".to_string()))
        }
    }

    /// Execute a SQL query
    pub fn execute_sql(&mut self, sql: &str) -> Result<Vec<Vec<SqliteValue>>, DatabaseError> {
        let table_name = Self::extract_table_name(sql)?;
        println!("Executing query for table: {}", table_name);
        self.execute_query(&table_name)
    }

    /// Get table structure
    fn get_table_info(&mut self, table_name: &str) -> Result<Table, DatabaseError> {
        let table_name = table_name.to_lowercase();
        // First, find the table in sqlite_master
        let schema_page = self.parse_btree_page(1)?;
        let page_data = self.read_page(1)?;

        for &cell_pointer in &schema_page.cell_pointers {
            if let Ok(record) = schema_page.get_cell_content(&page_data, cell_pointer) {
                if let (
                    SqliteValue::Text(type_name),
                    SqliteValue::Text(name),
                    SqliteValue::Integer(root_page),
                    SqliteValue::Text(_sql),
                ) = (
                    &record.values[0],
                    &record.values[1],
                    &record.values[3],
                    &record.values[4],
                ) {
                    if type_name.to_lowercase() == "table" && name.to_lowercase() == table_name {
                        return Ok(Table {
                            root_page: *root_page as u32,
                        });
                    }
                }
            }
        }

        Err(DatabaseError::TableNotFound(table_name))
    }

    /// Execute a simple SELECT query
    pub fn execute_query(
        &mut self,
        table_name: &str,
    ) -> Result<Vec<Vec<SqliteValue>>, DatabaseError> {
        let table = self.get_table_info(table_name)?;
        let mut results = Vec::new();

        // Read the table's root page
        let root_page = self.parse_btree_page(table.root_page)?;
        let page_data = self.read_page(table.root_page)?;

        // For now, we only handle leaf table pages
        if let PageType::LeafTable = root_page.page_type {
            for &cell_pointer in &root_page.cell_pointers {
                if let Ok(record) = root_page.get_cell_content(&page_data, cell_pointer) {
                    results.push(record.values);
                }
            }
        }

        Ok(results)
    }

    /// Traverse a b-tree table recursively
    fn traverse_table(
        &mut self,
        page_number: u32,
        where_clause: Option<&WhereClause>,
        results: &mut Vec<Vec<SqliteValue>>,
    ) -> Result<(), DatabaseError> {
        let page = self.parse_btree_page(page_number)?;
        let page_data = self.read_page(page_number)?;

        match page.page_type {
            PageType::LeafTable => {
                // Process all cells in the leaf page
                for &cell_pointer in &page.cell_pointers {
                    let cell = page.get_cell(&page_data, cell_pointer)?;
                    if let Ok(record) = Record::parse(&cell.payload) {
                        if let Some(where_clause) = where_clause {
                            if self.evaluate_where(&record.values, where_clause)? {
                                results.push(record.values);
                            }
                        } else {
                            results.push(record.values);
                        }
                    }
                }
            }
            PageType::InteriorTable => {
                // First, visit the left-most child
                if let Some(&first_pointer) = page.cell_pointers.first() {
                    let cell = page.get_cell(&page_data, first_pointer)?;
                    if let Some(left_child) = cell.left_child_page {
                        self.traverse_table(left_child, where_clause, results)?;
                    }
                }

                // Then process each cell and its right child
                for &cell_pointer in &page.cell_pointers {
                    let cell = page.get_cell(&page_data, cell_pointer)?;
                    // The row_id in the cell is the maximum row_id in the left subtree
                    // We need to visit the right child
                    if let Some(right_child) = cell.left_child_page {
                        self.traverse_table(right_child, where_clause, results)?;
                    }
                }
            }
            _ => {
                return {
                    println!("Unsupported page type: {:#?}", page.page_type);
                    Err(DatabaseError::InvalidPageType(page.page_type as u8))
                }
            }
        }

        Ok(())
    }

    fn evaluate_where(
        &self,
        values: &[SqliteValue],
        where_clause: &WhereClause,
    ) -> Result<bool, DatabaseError> {
        // Find the column index in the record
        let column_index = values
            .iter()
            .position(|_| true) // TODO: match column name with schema
            .ok_or_else(|| DatabaseError::InvalidColumn(where_clause.column.clone()))?;

        let record_value = &values[column_index];

        Ok(
            match (&record_value, &where_clause.operator, &where_clause.value) {
                (SqliteValue::Integer(a), Operator::Equals, SqliteValue::Integer(b)) => a == b,
                (SqliteValue::Integer(a), Operator::NotEquals, SqliteValue::Integer(b)) => a != b,
                (SqliteValue::Integer(a), Operator::GreaterThan, SqliteValue::Integer(b)) => a > b,
                (SqliteValue::Integer(a), Operator::LessThan, SqliteValue::Integer(b)) => a < b,
                (SqliteValue::Integer(a), Operator::GreaterEquals, SqliteValue::Integer(b)) => {
                    a >= b
                }
                (SqliteValue::Integer(a), Operator::LessEquals, SqliteValue::Integer(b)) => a <= b,

                (SqliteValue::Text(a), Operator::Equals, SqliteValue::Text(b)) => a == b,
                (SqliteValue::Text(a), Operator::NotEquals, SqliteValue::Text(b)) => a != b,
                (SqliteValue::Text(a), Operator::Like, SqliteValue::Text(b)) => {
                    // Simple LIKE implementation - only supports % at start/end
                    if b.starts_with('%') && b.ends_with('%') {
                        let pattern = &b[1..b.len() - 1];
                        a.contains(pattern)
                    } else if b.starts_with('%') {
                        let pattern = &b[1..];
                        a.ends_with(pattern)
                    } else if b.ends_with('%') {
                        let pattern = &b[..b.len() - 1];
                        a.starts_with(pattern)
                    } else {
                        a == b
                    }
                }
                _ => false,
            },
        )
    }

    /// Execute a query with optional WHERE clause
    pub fn execute_query_where(
        &mut self,
        table_name: &str,
        where_clause: Option<WhereClause>,
    ) -> Result<Vec<Vec<SqliteValue>>, DatabaseError> {
        let table = self.get_table_info(table_name)?;
        let mut results = Vec::new();

        self.traverse_table(table.root_page, where_clause.as_ref(), &mut results)?;

        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_db() {
        let db = Database::open("../resources/test.db");
        assert!(db.is_ok());
    }

    #[test]
    fn test_read_page() {
        let mut db = Database::open("../resources/test.db").unwrap();
        let page = db.read_page(1);
        assert!(page.is_ok());
    }

    #[test]
    fn test_parse_create_sql() {
        let sql = "CREATE TABLE test (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            age INTEGER,
            data BLOB
        )";

        let columns = Table::parse_create_sql(sql).unwrap();
        assert_eq!(columns.len(), 4);

        let id_column = &columns[0];
        assert_eq!(id_column.name, "id");
        assert_eq!(id_column.column_type, "INTEGER");
        assert!(id_column.primary_key);
        assert!(id_column.auto_increment);

        let name_column = &columns[1];
        assert_eq!(name_column.name, "name");
        assert_eq!(name_column.column_type, "TEXT");
        assert!(name_column.not_null);
    }

    #[test]
    fn test_where_clause() {
        let mut db = Database::open("../resources/test.db").unwrap();

        let where_clause = WhereClause {
            column: "name".to_string(),
            operator: Operator::Like,
            value: SqliteValue::Text("%John%".to_string()),
        };

        let results = db.execute_query_where("users", Some(where_clause));
        assert!(results.is_ok());
    }
}
