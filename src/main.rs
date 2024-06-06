// use csv::{ReaderBuilder, StringRecord};
use ndarray::{Array2, Array1, Axis, s};
use std::error::Error;
use linfa::dataset::Dataset;
use linfa::prelude::Fit;
use linfa_linear::LinearRegression;

fn contains_number(s: &&str) -> bool {
    s.chars().any(|c| c.is_digit(10))
}


fn read_data_file(path: &str) -> Result<(Vec<Vec<f64>>, Vec<String>), Box<dyn Error>> {
    let mut reader = csv::ReaderBuilder::new().delimiter(b';').from_path(path)?;
    let mut reader_temp = csv::ReaderBuilder::new().delimiter(b';').from_path(path)?;

    let mut rs: Vec<Vec<f64>> = Vec::new();
    let mut indexs: Vec<usize> = Vec::new();
    let mut dumy: Vec<Vec<String>> = Vec::new();

    // Process the header to extract feature names
    let headers = reader.headers()?.clone();
    let feature_names: Vec<String> = headers.iter().map(|s| s.split(',').
    collect::<Vec<&str>>()).flatten().map(|s| s.to_string()).collect();

    // Process the header to find categorical indices and collect unique values
    if let Some(result) = reader.records().next() {
        let record = result?;
        for (_index, field) in record.iter().enumerate() {
            let v: Vec<&str> = field.split(',').collect();
            for (i, f) in v.iter().enumerate() {
                if !contains_number(f) {
                    indexs.push(i);
                }
            }
        }

        for _ in indexs.iter() {
            let temp: Vec<String> = Vec::new();
            dumy.push(temp);
        }

        // Collect unique values for categorical fields
        for result in reader.records() {
            let record_1 = result?;
            for (_field_index, field) in record_1.iter().enumerate() {
                let v: Vec<&str> = field.split(',').collect();
                for (j, &index) in indexs.iter().enumerate() {
                    if !dumy[j].contains(&v.get(index).unwrap_or(&"").to_string()) {
                        dumy[j].push(v.get(index).unwrap_or(&"").to_string());
                    }
                }
            }
        }
        for j in &mut dumy {
            j.sort();
        }
            
    }
    for result in reader_temp.records() {
        let record = result?;
        let mut row: Vec<f64> = Vec::new();
        
        for field in &record {
            let v: Vec<&str> = field.split(',').collect();
            
            for vs in v {
                let mut found_index = None;
                // Check for categorical values
                for (_id, id_e) in dumy.iter().enumerate() {
                    for (idx, el) in id_e.iter().enumerate() {
                        if vs == el.as_str() {
                            found_index = Some(idx as f64);
                            break;
                        }
                    }
                    if found_index.is_some() {
                        break;
                    }
                }
                if let Some(idx) = found_index {
                    row.push(idx);
                } else {
                    if let Ok(num_value) = vs.parse::<f64>() {
                        row.push(num_value);
                    } else {
                        row.push(0.0);
                    }
                }
            }
        }
        rs.push(row);
    }
    
    Ok((rs, feature_names))
}


fn predict(data: Vec<Vec<f64>>) -> Result<(Array1<f64>, Array2<f64>), Box<dyn Error>> {
    // Convert data to Array2
    let rows = data.len();
    let cols = data.get(0).map_or(0, |row| row.len());
    let flattened_data: Vec<f64> = data.into_iter().flatten().collect();

    let array = Array2::from_shape_vec((rows, cols), flattened_data)?;

    let (data, targets) = (
        array.slice(s![.., 1..]).to_owned(),
        array.column(0).to_owned(),
    );

    // Create Dataset
    let dataset = Dataset::new(data.clone(), targets.clone())
        .with_feature_names(vec!["x"]);

    // Fit Linear Regression Model
    let lin_reg = LinearRegression::new();
    let model = lin_reg.fit(&dataset)?;

    // Making predictions
    let mut new_models: Vec<f64> = Vec::new();
    for row in data.axis_iter(Axis(0)) {
        let mut forecast = model.intercept();
        for (i, &value) in row.iter().enumerate() {
            forecast += model.params()[i] * value;
        }
        new_models.push(forecast);
    }
    let predictions = Array2::from_shape_vec((new_models.len(), 1), new_models)?;


    Ok((targets, predictions))
}


fn print_targets_and_predictions(targets: &Array1<f64>, predictions: &Array2<f64>) {
    assert_eq!(targets.len(), predictions.len(), "Targets and predictions must have the same length.");

    for (target, prediction) in targets.iter().zip(predictions.outer_iter()) {
        println!("Target: {:.2}, Prediction: {:.2}", target, prediction[0]);
    }
}


fn _r_squared(y_true: &Array1<f64>, y_pred: &Array2<f64>) -> f64 {
    // Calculate the mean of y_true
    let mean_y_true = y_true.mean().unwrap();
    
    // Calculate the total sum of squares (SS_tot)
    let ss_tot = y_true.iter().map(|_a| (_a - mean_y_true).powi(2)).sum::<f64>();
    
    // Flatten the y_pred array to treat all predictions as a single array
    let flattened_y_pred: Vec<f64> = y_pred.iter().cloned().collect();
    
    // Calculate the residual sum of squares (SS_res) for the entire set of predictions
    let ss_res = y_true.iter().zip(flattened_y_pred.iter()).map(|(a, b)| (mean_y_true - b).powi(2)).sum::<f64>();
    
    // Calculate R^2 for the entire dataset
    let r_squared = ss_res / ss_tot;
    
    r_squared
}


fn print_r_squared(_feature_names: &Vec<String>, r_squared_values: f64) {
    println!("R-squared: {}",r_squared_values * 100.0);
    // for (name, r_2) in feature_names.iter().zip(r_squared_values.iter()) {
    //     println!("{}: {:.2}%", name, r_2 * 100.0);
    // }
}


fn main() {
    if let Ok((data, feature_names)) = read_data_file("Housing.csv") {
        match predict(data) {
            Ok((targets, predictions)) => {
                print_targets_and_predictions(&targets, &predictions);
                let r_squared_values = _r_squared(&targets, &predictions);
                print_r_squared(
                    &feature_names,
                    r_squared_values,
                );
            }
            Err(err) => eprintln!("Error predicting: {}", err),
        }
    } else {
        eprintln!("Error reading file");
    }

    
}




