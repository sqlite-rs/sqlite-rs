use sqlite_rs::error::SqliteError;
use sqlite_rs::SqliteFile;

fn main() -> Result<(), SqliteError> {
    println!("Hello, world!");

    let mut sqlite = SqliteFile::open("./resources/test.db")?;

    println!("------ TABLES --------");
    let tables = sqlite.get_tables()?;
    for table in tables {
        println!("Table: {:?}", table);
    }

    println!("------ EXEC SQL --------");
    let res = sqlite.execute_sql("SELECT * FROM t1")?;
    println!("{:?}", res);

    Ok(())
}
