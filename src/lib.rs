use std::collections::BTreeSet;
use ndarray::{indices, s, Array1, Array2, Axis};
use std::error::Error;
use linfa::dataset::Dataset;
use linfa::prelude::Fit;
use linfa_linear::LinearRegression;

pub fn contains_number(s: &str) -> bool {
    s.chars().any(|c| c.is_digit(10))
}
pub fn parse_data(input: Vec<Vec<String>>) -> Result<(Vec<Vec<f64>>, Vec<String>), Box<dyn Error>> {
    let mut rs: Vec<Vec<f64>> = Vec::new();
    let mut indexs: Vec<usize> = Vec::new();
    let mut dumy: Vec<BTreeSet<String>> = Vec::new();
        let feature_names = input[0].clone();

    // Determine which columns contain strings
    for (idx, vl) in input[1].iter().enumerate() {
        if !contains_number(vl.as_str()) {
            indexs.push(idx);
        }
    }

    for _ in indexs.iter() {
        dumy.push(BTreeSet::new());
    }

    // Populate dumy with unique strings from the identified columns
    for i in 1..input.len() {
        let v = &input[i];
        for (j, &idx) in indexs.iter().enumerate() {
            if let Some(val) = v.get(idx) {
                dumy[j].insert(val.clone());
            }
        }
    }
    

    println!("{:?}",dumy);
    // Process each row in input to create the result vector
    for i in 1..input.len() {
        let v = &input[i];
        if v.len() == input[0].len() {
            let mut result_row: Vec<f64> = Vec::new();
        for (idx, vl) in v.iter().enumerate() {
            if let Some(pos) = indexs.iter().position(|&i| i == idx) {
                // If the column contains strings, find the index of the string in dumy
                if let Some(dumy_idx) = dumy[pos].iter().position(|s| s == vl) {
                    result_row.push(dumy_idx as f64);
                } else {
                    result_row.push(-1.0);
                }
            } else {
                // Otherwise, parse it as f64
                if let Ok(parsed_value) = vl.parse::<f64>() {
                    result_row.push(parsed_value);
                } else {
                    result_row.push(0.0);
                }
            }
        }
        rs.push(result_row);
        }
    }
    Ok((rs, feature_names))
}




pub fn predict(data: Vec<Vec<f64>>) -> Result<(Array1<f64>, Array2<f64>), Box<dyn Error>> {

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




pub fn print_targets_and_predictions(targets: &Array1<f64>, predictions: &Array2<f64>) {
    assert_eq!(targets.len(), predictions.len(), "Targets and predictions must have the same length.");

    for (target, prediction) in targets.iter().zip(predictions.outer_iter()) {
        println!("Target: {:.2}, Prediction: {:.2}", target, prediction[0]);
    }
}


pub fn _r_squared(y_true: &Array1<f64>, y_pred: &Array2<f64>) -> f64 {
    // Calculate the mean of y_true
    let mean_y_true = y_true.mean().unwrap();
    
    // Calculate the total sum of squares (SS_tot)
    let ss_tot = y_true.iter().map(|_a| (_a - mean_y_true).powi(2)).sum::<f64>();
    
    // Flatten the y_pred array to treat all predictions as a single array
    let flattened_y_pred: Vec<f64> = y_pred.iter().cloned().collect();
    
    // Calculate the residual sum of squares (SS_res) for the entire set of predictions
    let ss_res = y_true.iter().zip(flattened_y_pred.iter()).map(|(_a, b)| (mean_y_true - b).powi(2)).sum::<f64>();
    
    // Calculate R^2 for the entire dataset
    let r_squared = ss_res / ss_tot;
    
    r_squared
}


pub fn print_r_squared(_feature_names: &Vec<String>, r_squared_values: f64) {
    println!("R-squared: {}",r_squared_values * 100.0);

}
