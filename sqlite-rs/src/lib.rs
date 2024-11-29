pub mod error;

use crate::error::SqliteError;
use byteorder::{BigEndian, ReadBytesExt};
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug)]
pub struct TableInfo {
    pub name: String,
    pub sql: String,
    pub root_page: u32,
}

#[derive(Debug)]
pub struct Column {
    pub name: String,
    pub type_name: String,
}

#[derive(Debug)]
pub struct Value {
    pub data: Vec<u8>,
    pub type_code: u8,
}

#[derive(Debug)]
pub struct Row {
    pub values: Vec<Value>,
}

#[derive(Debug)]
pub struct QueryResult {
    pub columns: Vec<Column>,
    pub rows: Vec<Row>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PageType {
    InteriorIndex = 0x02,
    InteriorTable = 0x05,
    LeafIndex = 0x0A,
    LeafTable = 0x0D,
}

impl TryFrom<u8> for PageType {
    type Error = SqliteError;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0x02 => Ok(PageType::InteriorIndex),
            0x05 => Ok(PageType::InteriorTable),
            0x0A => Ok(PageType::LeafIndex),
            0x0D => Ok(PageType::LeafTable),
            _ => Err(SqliteError::InvalidPageType),
        }
    }
}

pub struct SqliteFile {
    file: File,
    page_size: u16,
}

impl SqliteFile {
    pub fn open(path: &str) -> Result<Self, SqliteError> {
        let mut file = File::open(path)?;
        // 验证 SQLite 文件头
        let mut header = [0u8; 16];
        file.read_exact(&mut header)?;

        // SQLite 文件头应该以 "SQLite format 3\0" 开始
        if &header[0..15] != b"SQLite format 3" || header[15] != 0 {
            return Err(SqliteError::InvalidHeader);
        }

        // 读取数据库页大小（2字节）
        let page_size = file.read_u16::<BigEndian>()?;

        // 读取文件格式版本
        file.seek(SeekFrom::Start(18))?;
        let write_version = file.read_u8()?;
        let read_version = file.read_u8()?;
        println!(
            "write version: {}, read version: {}",
            write_version, read_version
        );

        if write_version != 1 || read_version != 1 {
            return Err(SqliteError::UnsupportedVersion);
        }

        Ok(SqliteFile { file, page_size })
    }

    pub fn read_page(&mut self, page_number: u32) -> Result<Vec<u8>, SqliteError> {
        let offset = (page_number as u64 - 1) * self.page_size as u64;
        self.file.seek(SeekFrom::Start(offset))?;

        let mut buffer = vec![0; self.page_size as usize];
        self.file.read_exact(&mut buffer)?;

        Ok(buffer)
    }

    pub fn get_tables(&mut self) -> Result<Vec<TableInfo>, SqliteError> {
        // sqlite_master 表总是在第一页
        let page_data = self.read_page(1)?;

        // SQLite 文件头是 100 字节
        let header_size = 100;
        // 从文件头后开始读取页面类型
        let page_type = page_data[header_size];
        if page_type != 0x0D {
            // 0x0D 表示叶子表页
            return Err(SqliteError::InvalidPageType);
        }

        let cell_count =
            u16::from_be_bytes([page_data[header_size + 3], page_data[header_size + 4]]) as usize;
        println!("Cell count: {}", cell_count);

        let mut tables = Vec::new();
        // 遍历所有单元格
        for i in 0..cell_count {
            let cell_offset_start = header_size + 8 + i * 2;
            let cell_offset = u16::from_be_bytes([
                page_data[cell_offset_start],
                page_data[cell_offset_start + 1],
            ]) as usize;

            // 解析单元格内容
            let record = self.parse_cell(&page_data[cell_offset..])?;
            println!("Record: {:02X?}", record);
            if let Some(table_info) = self.extract_table_info(&record) {
                tables.push(table_info);
            }
        }

        Ok(tables)
    }

    fn parse_cell(&self, cell_data: &[u8]) -> Result<Vec<Vec<u8>>, SqliteError> {
        println!("Cell data length: {}", cell_data.len());
        println!("Cell data: {:02X?}", cell_data);

        let mut pos = 0;

        // 检查数据长度
        if pos >= cell_data.len() {
            return Err(SqliteError::InvalidCellContent);
        }

        let payload_size = self.parse_varint(cell_data, &mut pos)?;
        println!("Payload size: {}, pos after: {}", payload_size, pos);

        let _rowid = self.parse_varint(cell_data, &mut pos)?;
        println!("Row ID: {}, pos after: {}", _rowid, pos);

        let header_size = self.parse_varint(cell_data, &mut pos)? as usize;
        println!("Header size: {}, pos after: {}", header_size, pos);

        // 读取列类型数组
        let header_start = pos;
        let header_end = header_start + header_size;
        println!("Header bytes: {:02X?}", &cell_data[pos..header_end]);

        if header_end > cell_data.len() {
            println!(
                "Invalid header size: pos({}) + header_size({}) > len({})",
                pos, header_size, cell_data.len()
            );
            return Err(SqliteError::InvalidCellContent);
        }

        let mut serial_types = Vec::new();
        let mut header_pos = header_start;
        while header_pos < header_end {
            let serial_type = self.parse_varint(cell_data, &mut header_pos)?;
            println!("Serial type: {}", serial_type);
            serial_types.push(serial_type);
        }
        println!("Serial types: {:?}", serial_types);

        pos = header_end;
        let mut values = Vec::new();

        // 根据类型解析每个值
        for serial_type in serial_types {
            let value = match serial_type {
                0 => Vec::new(), // NULL
                1 => {
                    if pos >= cell_data.len() {
                        return Err(SqliteError::InvalidCellContent);
                    }
                    let v = vec![cell_data[pos]];
                    pos += 1;
                    v
                } // 8-bit signed int
                2 => {
                    if pos + 2 > cell_data.len() {
                        return Err(SqliteError::InvalidCellContent);
                    }
                    let v = cell_data[pos..pos + 2].to_vec();
                    pos += 2;
                    v
                } // 16-bit signed int
                3 => {
                    if pos + 3 > cell_data.len() {
                        return Err(SqliteError::InvalidCellContent);
                    }
                    let v = cell_data[pos..pos + 3].to_vec();
                    pos += 3;
                    v
                } // 24-bit signed int
                4 => {
                    if pos + 4 > cell_data.len() {
                        return Err(SqliteError::InvalidCellContent);
                    }
                    let v = cell_data[pos..pos + 4].to_vec();
                    pos += 4;
                    v
                } // 32-bit signed int
                5 => {
                    if pos + 6 > cell_data.len() {
                        return Err(SqliteError::InvalidCellContent);
                    }
                    let v = cell_data[pos..pos + 6].to_vec();
                    pos += 6;
                    v
                } // 48-bit signed int
                6 => {
                    if pos + 8 > cell_data.len() {
                        return Err(SqliteError::InvalidCellContent);
                    }
                    let v = cell_data[pos..pos + 8].to_vec();
                    pos += 8;
                    v
                } // 64-bit signed int
                7 => {
                    if pos + 8 > cell_data.len() {
                        return Err(SqliteError::InvalidCellContent);
                    }
                    let v = cell_data[pos..pos + 8].to_vec();
                    pos += 8;
                    v
                } // 64-bit IEEE float
                8 | 9 => Vec::new(), // 常量 0 和 1
                n if n >= 12 => {
                    // 计算字符串长度
                    let size = if n % 2 == 0 {
                        ((n - 12) / 2) as usize
                    } else {
                        ((n - 13) / 2) as usize
                    };

                    if size > 0 {
                        if pos + size > cell_data.len() {
                            println!(
                                "Invalid string size: pos({}) + size({}) > len({}), serial_type:{}",
                                pos,
                                size,
                                cell_data.len(),
                                n
                            );
                            return Err(SqliteError::InvalidCellContent);
                        }

                        let value = cell_data[pos..pos + size].to_vec();
                        println!("String value (size {}): {:02X?}", size, value);
                        pos += size;
                        value
                    } else {
                        Vec::new()
                    }
                }
                _ => {
                    println!("Unknown serial type: {}", serial_type);
                    return Err(SqliteError::InvalidCellContent);
                }
            };

            println!("Value: {:02X?}", value);
            values.push(value);
        }

        Ok(values)
    }

    fn extract_table_info(&self, record: &[Vec<u8>]) -> Option<TableInfo> {
        if record.len() >= 5 {
            let type_str = String::from_utf8_lossy(&record[0]);
            let name = String::from_utf8_lossy(&record[1]);
            let tbl_name = String::from_utf8_lossy(&record[2]);
            let rootpage = record[3][0] as u32;
            let sql = String::from_utf8_lossy(&record[4]);

            println!("Table record:");
            println!("  Type: {}", type_str);
            println!("  Name: {}", name);
            println!("  Table Name: {}", tbl_name);
            println!("  Root Page: {}", rootpage);
            println!("  SQL: {}", sql);

            Some(TableInfo {
                name: name.to_string(),
                root_page: rootpage,
                sql: sql.to_string(),
            })
        } else {
            None
        }
    }

    fn parse_varint(&self, data: &[u8], pos: &mut usize) -> Result<u64, SqliteError> {
        let start_pos = *pos;

        // 检查数据长度
        if *pos >= data.len() {
            return Err(SqliteError::InvalidCellContent);
        }

        let first_byte = data[*pos];
        *pos += 1;

        if first_byte <= 0x7F {
            // 单字节，直接返回
            return Ok(first_byte as u64);
        }

        let mut result = (first_byte & 0x7F) as u64;
        let mut shift = 7;

        for i in 1..9 {
            if *pos >= data.len() {
                println!(
                    "Invalid varint: pos({}) + i({}) > len({})",
                    start_pos,
                    i,
                    data.len()
                );
                return Err(SqliteError::InvalidCellContent);
            }

            let byte = data[*pos];
            *pos += 1;

            if i == 8 {
                // 最后一个字节，直接使用整个字节
                result |= (byte as u64) << shift;
                break;
            } else {
                // 其他字节，使用低 7 位
                result |= ((byte & 0x7F) as u64) << shift;
                if byte & 0x80 == 0 {
                    // 最高位为 0，表示结束
                    break;
                }
                shift += 7;
            }
        }

        // 如果超过 9 字节，说明格式错误
        if *pos - start_pos > 9 {
            println!("Invalid varint: pos({}) + i({})", start_pos, 9,);
            return Err(SqliteError::InvalidCellContent);
        }

        Ok(result)
    }

    pub fn execute_sql(&mut self, sql: &str) -> Result<QueryResult, SqliteError> {
        // 简单的 SQL 解析，目前只支持 SELECT * FROM table_name
        let sql = sql.trim().to_lowercase();
        if !sql.starts_with("select") {
            return Err(SqliteError::SqlParseError(
                "Only SELECT queries are supported".to_string(),
            ));
        }

        let parts: Vec<&str> = sql.split_whitespace().collect();
        if parts.len() < 4 || parts[1] != "*" || parts[2] != "from" {
            return Err(SqliteError::SqlParseError(
                "Invalid SELECT query format".to_string(),
            ));
        }

        let table_name = parts[3];
        self.read_table(table_name)
    }

    fn read_table(&mut self, table_name: &str) -> Result<QueryResult, SqliteError> {
        // 获取表信息
        let tables = self.get_tables()?;
        let table = tables
            .iter()
            .find(|t| t.name == table_name)
            .ok_or_else(|| SqliteError::TableNotFound(table_name.to_string()))?;

        // 解析表结构
        let columns = self.parse_table_schema(&table.sql)?;

        // 读取表数据页
        let mut rows = Vec::new();
        self.read_table_page(table.root_page, &mut rows)?;

        Ok(QueryResult { columns, rows })
    }

    fn read_table_page(
        &mut self,
        page_number: u32,
        rows: &mut Vec<Row>,
    ) -> Result<(), SqliteError> {
        println!("Reading page number: {}", page_number);
        let page_data = self.read_page(page_number)?;

        // 打印页面头部数据用于调试
        println!("Page header bytes: {:02X?}", &page_data[0..16]);

        // 检查页类型
        let page_type = page_data[0];
        println!("Page type byte: 0x{:02X}", page_type);

        // 特殊处理第一页，它可能是文件头
        if page_number == 1 {
            // 第一页有100字节的文件头
            let header_size = 100;

            // 获取第一个页面的实际内容的偏移量
            let content_offset = header_size;

            // 从实际内容开始读取页面类型
            let actual_page_type = page_data[content_offset];
            println!("First page actual type byte: 0x{:02X}", actual_page_type);

            let page_type = PageType::try_from(actual_page_type).map_err(|_| {
                println!("Invalid page type: 0x{:02X}", actual_page_type);
                SqliteError::InvalidPageType
            })?;

            match page_type {
                PageType::LeafTable => {
                    // 读取单元格数量（偏移量需要考虑文件头）
                    let cell_count = u16::from_be_bytes([
                        page_data[content_offset + 3],
                        page_data[content_offset + 4],
                    ]) as usize;
                    println!("Cell count: {}", cell_count);

                    // 读取每个单元格
                    for i in 0..cell_count {
                        let cell_offset_start = content_offset + 8 + i * 2;
                        let cell_offset = u16::from_be_bytes([
                            page_data[cell_offset_start],
                            page_data[cell_offset_start + 1],
                        ]) as usize;

                        let record = self.parse_cell(&page_data[cell_offset..])?;
                        rows.push(Row {
                            values: record
                                .into_iter()
                                .map(|data| Value { data, type_code: 0 })
                                .collect(),
                        });
                    }
                }
                PageType::InteriorTable => {
                    // 读取单元格数量
                    let cell_count = u16::from_be_bytes([
                        page_data[content_offset + 3],
                        page_data[content_offset + 4],
                    ]) as usize;
                    println!("Cell count: {}", cell_count);

                    // 读取右子页面
                    let right_child = u32::from_be_bytes([
                        page_data[content_offset + 8],
                        page_data[content_offset + 9],
                        page_data[content_offset + 10],
                        page_data[content_offset + 11],
                    ]);
                    self.read_table_page(right_child, rows)?;

                    // 读取其他子页面
                    for i in 0..cell_count {
                        let cell_offset_start = content_offset + 12 + i * 2;
                        let cell_offset = u16::from_be_bytes([
                            page_data[cell_offset_start],
                            page_data[cell_offset_start + 1],
                        ]) as usize;

                        // 从单元格中读取左子页面号
                        let child_page = u32::from_be_bytes([
                            page_data[cell_offset],
                            page_data[cell_offset + 1],
                            page_data[cell_offset + 2],
                            page_data[cell_offset + 3],
                        ]);
                        self.read_table_page(child_page, rows)?;
                    }
                }
                _ => {
                    println!("Unsupported page type for first page: {:?}", page_type);
                    return Err(SqliteError::InvalidPageType);
                }
            }
        } else {
            // 非第一页的处理逻辑
            let page_type = PageType::try_from(page_type).map_err(|_| {
                println!("Invalid page type: 0x{:02X}", page_type);
                SqliteError::InvalidPageType
            })?;

            match page_type {
                PageType::LeafTable => {
                    let cell_count = u16::from_be_bytes([page_data[3], page_data[4]]) as usize;
                    println!("Cell count: {}", cell_count);

                    for i in 0..cell_count {
                        let cell_offset_start = 8 + i * 2;
                        let cell_offset = u16::from_be_bytes([
                            page_data[cell_offset_start],
                            page_data[cell_offset_start + 1],
                        ]) as usize;

                        let record = self.parse_cell(&page_data[cell_offset..])?;
                        rows.push(Row {
                            values: record
                                .into_iter()
                                .map(|data| Value { data, type_code: 0 })
                                .collect(),
                        });
                    }
                }
                PageType::InteriorTable => {
                    let cell_count = u16::from_be_bytes([page_data[3], page_data[4]]) as usize;
                    println!("Cell count: {}", cell_count);

                    let right_child = u32::from_be_bytes([
                        page_data[8],
                        page_data[9],
                        page_data[10],
                        page_data[11],
                    ]);
                    self.read_table_page(right_child, rows)?;

                    for i in 0..cell_count {
                        let cell_offset_start = 12 + i * 2;
                        let cell_offset = u16::from_be_bytes([
                            page_data[cell_offset_start],
                            page_data[cell_offset_start + 1],
                        ]) as usize;

                        let child_page = u32::from_be_bytes([
                            page_data[cell_offset],
                            page_data[cell_offset + 1],
                            page_data[cell_offset + 2],
                            page_data[cell_offset + 3],
                        ]);
                        self.read_table_page(child_page, rows)?;
                    }
                }
                _ => {
                    println!("Unsupported page type: {:?}", page_type);
                    return Err(SqliteError::InvalidPageType);
                }
            }
        }

        Ok(())
    }

    fn parse_table_schema(&self, sql: &str) -> Result<Vec<Column>, SqliteError> {
        // 简单的 CREATE TABLE 语句解析
        let start = sql
            .find('(')
            .ok_or_else(|| SqliteError::SqlParseError("Invalid CREATE TABLE syntax".to_string()))?;
        let end = sql
            .rfind(')')
            .ok_or_else(|| SqliteError::SqlParseError("Invalid CREATE TABLE syntax".to_string()))?;

        let column_defs = &sql[start + 1..end];
        let columns = column_defs
            .split(',')
            .filter_map(|col| {
                let parts: Vec<&str> = col.trim().split_whitespace().collect();
                if parts.len() >= 2 {
                    Some(Column {
                        name: parts[0].to_string(),
                        type_name: parts[1].to_string(),
                    })
                } else {
                    None
                }
            })
            .collect();

        Ok(columns)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_invalid_file() {
        let result = SqliteFile::open("nonexistent.db");
        assert!(result.is_err());
    }
}
