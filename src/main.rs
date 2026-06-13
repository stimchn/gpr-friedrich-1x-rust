use plotters::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rdr = csv::ReaderBuilder::new()
        .has_headers(true)
        .from_path("train_data.csv")?;

    let mut x_train: Vec<Vec<f64>> = Vec::new();
    let mut y_train: Vec<f64> = Vec::new();

    for result in rdr.records() {
        let record = result?;
        let x: f64 = record[0].parse()?;
        let y: f64 = record[1].parse()?;
        x_train.push(vec![x]);
        y_train.push(y);
    }

    let xy_train: Vec<(f64, f64)> = x_train
        .iter()
        .zip(y_train.iter())
        .map(|(x, y)| (x[0], *y))
        .collect();
    println!("xy_train = {:?}", xy_train);

    let gp_model = friedrich::gaussian_process::GaussianProcess::default(x_train, y_train);
    println!("gp_model.kernel = {:?}", gp_model.kernel);

    let x_test: Vec<Vec<f64>> = (-20..=120).map(|i| vec![i as f64 * 0.01]).collect();
    let y_mu = gp_model.predict(&x_test);
    let y_var = gp_model.predict_variance(&x_test);
    for i in 0..x_test.len() {
        println!(
            "{:.2}, {:>12.6}, {:12.6}",
            x_test[i][0],
            y_mu[i],
            y_var[i].sqrt()
        );
    }

    let xy_test: Vec<(f64, f64, f64)> = x_test
        .iter()
        .zip(y_mu.iter())
        .zip(y_var.iter())
        .map(|((x, y), s)| (x[0], *y, (*s).sqrt()))
        .collect();

    let root = BitMapBackend::new("image.png", (3840, 2160)).into_drawing_area();
    root.fill(&WHITE)?;
    let mut chart = ChartBuilder::on(&root)
        .caption("GPR Test (friedrich)", ("sans-serif", 128).into_font())
        .margin(100)
        .x_label_area_size(200)
        .y_label_area_size(200)
        .build_cartesian_2d(-0.2f64..1.2f64, -2.5f64..2.5f64)?;

    chart
        .configure_mesh()
        .x_desc("x")
        .y_desc("y")
        .axis_desc_style(("sans-serif", 96))
        .label_style(("sans-serif", 64))
        .draw()?;

    chart.draw_series(
        xy_train
            .iter()
            .map(|(x, y)| Circle::new((*x, *y), 16, WHITE.filled())),
    )?;
    chart
        .draw_series(xy_train.iter().map(|(x, y)| {
            Circle::new(
                (*x, *y),
                16,
                ShapeStyle {
                    color: BLACK.to_rgba(),
                    filled: false,
                    stroke_width: 4,
                },
            )
        }))?
        .label("Train data")
        .legend(|(x, y)| {
            EmptyElement::at((x, y))
                + Circle::new((10, 0), 8, WHITE.filled())
                + Circle::new(
                    (10, 0),
                    8,
                    ShapeStyle {
                        color: BLACK.to_rgba(),
                        filled: false,
                        stroke_width: 4,
                    },
                )
        });

    chart
        .draw_series(DashedLineSeries::new(
            xy_test.iter().map(|(x, y, s)| (*x, *y + 3.0 * *s)),
            10,
            10,
            BLACK.stroke_width(4),
        ))?
        .label("μ+3σ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(4)));
    chart
        .draw_series(LineSeries::new(
            xy_test.iter().map(|(x, y, _)| (*x, *y)),
            BLACK.stroke_width(4),
        ))?
        .label("μ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], BLACK.stroke_width(4)));
    chart
        .draw_series(DashedLineSeries::new(
            xy_test.iter().map(|(x, y, s)| (*x, *y - 3.0 * *s)),
            10,
            10,
            BLACK.stroke_width(4),
        ))?
        .label("μ-3σ")
        .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 30, y)], BLACK.stroke_width(4)));

    chart
        .configure_series_labels()
        .position(SeriesLabelPosition::UpperLeft)
        .margin(32)
        .label_font(("sans-serif", 64))
        .background_style(&WHITE.mix(0.8))
        .border_style(&BLACK)
        .draw()?;

    root.present()?;

    Ok(())
}
