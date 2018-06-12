extern crate gnuplot;
extern crate rand;

use gnuplot::{Color, Figure};
use rand::{seq, thread_rng};
use std::collections::HashMap;
use std::collections::VecDeque;
use std::f64;

#[derive(Debug, Clone)]
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Point {
        Point { x, y }
    }
}

fn all_close(a: &Vec<f64>, b: &Vec<f64>) -> bool {
    let rtol: f64 = 1e-05;
    let atol: f64 = 1e-08;
    let mut result = true;
    assert_eq!(a.len(), b.len());
    let it = a.iter().zip(b.iter());
    for (a, b) in it {
        if (a - b).abs() > (atol + rtol * b.abs()) {
            result = false;
        }
    }
    result
}

fn count_distance(points: &Vec<Point>, centroid: &Vec<Point>) -> (Vec<Vec<f64>>, f64) {
    let mut distance = Vec::<Vec<f64>>::new();
    let mut dist = 0f64;
    for c in centroid {
        let mut d = Vec::<f64>::new();
        for p in points {
            let c_p_dist = (((p.x - c.x).powi(2)) + (p.y - c.y).powi(2)).sqrt();
            dist += c_p_dist;
            d.push(c_p_dist);
        }
        distance.push(d);
    }
    (distance, dist)
}

fn argmin(dist: Vec<Vec<f64>>, n_clusters: usize, points_len: usize) -> Vec<usize> {
    let mut cluster = Vec::<usize>::with_capacity(n_clusters);
    for c in 0..points_len {
        let mut max = f64::MAX;
        let mut idx: usize = 0;
        for (i, d) in dist.iter().enumerate() {
            if d[c] < max {
                idx = i;
                max = d[c];
            }
        }
        cluster.push(idx);
    }
    cluster
}

fn update_centroids(
    centroid: &mut Vec<Point>,
    points: &Vec<Point>,
    cluster: &Vec<usize>,
    n_clusters: usize,
) {
    let mut mean = HashMap::<usize, ((f64, f64), f64)>::new();
    for c in 0..n_clusters {
        mean.insert(c, ((0., 0.), 0.));
    }
    for (i, p) in points.iter().enumerate() {
        let m = mean.get_mut(&cluster[i]).unwrap();
        let x = (m.0).0 + p.x;
        let y = (m.0).1 + p.y;
        let count = m.1 + 1.;
        *m = ((x, y), count);
    }
    for (k, v) in mean {
        centroid[k] = Point {
            x: (v.0).0 / v.1,
            y: (v.0).1 / v.1,
        };
    }
}

fn kmeans(points: &Vec<Point>, n_clusters: usize) -> Vec<usize> {
    let mut centroid = Vec::<Point>::with_capacity(n_clusters);
    let mut rng = thread_rng();
    let mut cluster = vec![];
    let sample = seq::sample_iter(&mut rng, 0..points.len(), n_clusters).unwrap();

    for i in &sample {
        centroid.push(points[*i].clone());
    }
    let mut loss = VecDeque::<Vec<f64>>::new();
    loss.push_front(vec![-1.]);
    loss.push_front(vec![-2.]);
    while !all_close(&loss[0], &loss[1]) {
        let (distance, dist) = count_distance(points, &centroid);
        loss.pop_front();
        loss.push_back(vec![dist]);
        cluster = argmin(distance, n_clusters, points.len());
        update_centroids(&mut centroid, &points, &cluster, n_clusters);
    }
    cluster
}

fn main() {
    let sample_point = 1000;
    let number_of_cluster = 5;
    let mut points = Vec::<Point>::with_capacity(sample_point);
    let mut cluster_point = Vec::<(Vec<f64>, Vec<f64>)>::new();
    for _ in 0..sample_point {
        let x = rand::random::<f64>() % 1000.;
        let y = rand::random::<f64>() % 1000.;
        points.push(Point::new(x, y));
    }
    for _ in 0..number_of_cluster {
        cluster_point.push((Vec::<f64>::new(), Vec::<f64>::new()));
    }
    let cluster = kmeans(&points, number_of_cluster);

    for (i, p) in points.iter().enumerate() {
        let c = cluster[i];
        cluster_point[c].0.push(p.x);
        cluster_point[c].1.push(p.y);
    }
    let color = ["red", "blue", "green", "yellow", "black"];
    let mut fg = Figure::new();
    fg.axes2d()
        .points(&cluster_point[0].0, &cluster_point[0].1, &[Color(color[0])])
        .points(&cluster_point[1].0, &cluster_point[1].1, &[Color(color[1])])
        .points(&cluster_point[2].0, &cluster_point[2].1, &[Color(color[2])])
        .points(&cluster_point[3].0, &cluster_point[3].1, &[Color(color[3])])
        .points(&cluster_point[4].0, &cluster_point[4].1, &[Color(color[4])]);

    fg.show();
    println!("{:?}", cluster);
}
