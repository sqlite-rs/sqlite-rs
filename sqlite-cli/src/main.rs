use sqlite_rs::Database;

fn main() {
    let mut db = Database::open("resources/test.db").expect("Failed to open database");
    
    // List all tables
    match db.list_tables() {
        Ok(tables) => {
            println!("Tables in database:");
            for table in tables {
                println!("{}", table);
            }
        }
        Err(e) => println!("Error listing tables: {:?}", e),
    }

    // Execute a query
    match db.execute_sql("SELECT * FROM t1") {
        Ok(rows) => {
            println!("\nQuery results:");
            for row in rows {
                println!("{:?}", row);
            }
        }
        Err(e) => println!("Error executing query: {:?}", e),
    }
}
