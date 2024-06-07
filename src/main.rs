mod lib;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let response = reqwest::get("https://a-simple-demo.spore.pro/api/media/0xed725757bb5f3d74574c0a5784cca0706e5935943f78cd97382a31c3f9dc10ed").await?;
    let json_response: Vec<Vec<String>> = response.json().await?;
    match lib::parse_data(json_response) {
        Ok((data, feature_names)) => {
            match lib::predict(data) {
                Ok((targets, predictions)) =>{
                    lib::print_targets_and_predictions(&targets, &predictions);
                    lib::print_r_squared(&feature_names, lib::_r_squared(&targets, &predictions));
                }
                Err(e) =>{
                    eprintln!("Error: {:?}",e);
                }
            }
        }
        Err(e) => {
            eprintln!("Error: {:?}", e);
        }
    }
    
    Ok(())
}
