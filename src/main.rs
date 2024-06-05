use csv::ReaderBuilder;
use ndarray::{Array2, s};
use std::error::Error;
use linfa::dataset::Dataset;
use linfa::traits::Transformer;
use linfa::prelude::LinearRegression;



fn leer_archivo(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let mut reader = ReaderBuilder::new().delimiter(b';').from_path(path)?;
    let mut rs: Vec<Vec<f64>> = Vec::new();

    for result in reader.records() {
        let record = result?;
        for field in &record {
            let v: Vec<&str> = field.split(',').collect();
            let mut demo: Vec<f64> = Vec::new();
            for vs in v {
                let converted_value = match vs {
                    "yes" => 1.0,
                    "no" => 0.0,
                    "furnished" => 0.0,
                    "semi-furnished" => 1.0,
                    "unfurnished" => 2.0,
                    _ => vs.parse::<f64>().unwrap_or(0.0),  // Default to 0.0 if parsing fails
                };
                demo.push(converted_value);
            }
            rs.push(demo);
        }
    }
    Ok(rs)
}

fn main() -> Result<(), Box<dyn Error>> {
    let data = leer_archivo("Housing.csv")?;

    // Determine the number of rows and columns
    let rows = data.len();
    let cols = data.get(0).map_or(0, |row| row.len());

    // Flatten the vector of vectors into a single vector
    let flattened_data: Vec<f64> = data.into_iter().flatten().collect();

    // Create the Array2
    let array = Array2::from_shape_vec((rows, cols), flattened_data)?;

    // Print the Array2
    // println!("{:?}", array);
    
    let (data, targets) = (
        array.slice(s![.., 1..]).to_owned(),
        array.column(0).to_owned(),
    );
    // println!("{:?}",targets);
    // println!("{:?}",data);
    
    let x_max = data.iter().cloned().fold(f64::MIN, f64::max).ceil();
    let y_max = targets.iter().cloned().fold(f64::MIN, f64::max).ceil();

    // println!("x_max: {:?}", x_max);
    // println!("y_max: {:?}", y_max);

    let dataset = Dataset::new(data, targets).
    with_feature_names(vec!["area", "bedrooms","bathrooms","stories","mainroad","guestroom","basement",
                            "hotwaterheating","airconditioning","parking","prefarea","furnishingstatus"]);

    // Split dataset into features (X) and target (y)
    let X = dataset.records();
    let y = dataset.targets().into_owned();

    // Split into training and testing sets (e.g., 80% train, 20% test)
    let split_point = (X.nrows() as f64 * 0.8) as usize;
    let (X_train, X_test) = X.select(ndarray::s![0..split_point, split_point..]);
    let (y_train, y_test) = (y.select(Axis(0), &0..split_point), y.select(Axis(0), &split_point..));
    



    Ok(())
}
