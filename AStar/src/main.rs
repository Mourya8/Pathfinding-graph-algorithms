
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::path::Path;
use std::cmp::Ordering;
use std::collections::BinaryHeap;

// Graph structure
#[derive(Debug)]
struct Graph {
    adjacency_matrix: Vec<Vec<i32>>,
    node_weights: Vec<Vec<i32>>, 
}

// Priority queue item for A*
#[derive(Debug)]
struct State {
    cost: i32,   
    node: usize,
}


impl Ord for State {
    fn cmp(&self, other: &Self) -> Ordering {
        other.cost.cmp(&self.cost) 
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Eq for State {}

// Function to read the graph from a file
fn read_graph_from_file(filename: &str) -> io::Result<Graph> {
    let path = Path::new(filename);
    let file = File::open(&path)?;
    let mut lines = BufReader::new(file).lines();

    let first_line = lines.next().unwrap()?;
    let mut parts = first_line.split_whitespace();
    let n: usize = parts.next().unwrap().parse().unwrap();
    let _m: usize = parts.next().unwrap().parse().unwrap();

    let mut adjacency_matrix = vec![vec![0; n]; n];
    for i in 0..n {
        let line = lines.next().unwrap()?;
        adjacency_matrix[i] = line
            .split_whitespace()
            .map(|x| x.parse::<i32>().unwrap())
            .collect();
    }

    let mut node_weights = Vec::new();
    while let Some(line) = lines.next() {
        let weights: Vec<i32> = line?
            .split_whitespace()
            .map(|x| x.parse::<i32>().unwrap())
            .collect();
        node_weights.push(weights);
    }

    Ok(Graph {
        adjacency_matrix,
        node_weights,
    })
}


fn heuristic(_graph: &Graph, _node: usize, _goal: usize) -> i32 {
    0 
}

// A* Pathfinding (Serial)
fn a_star_serial(graph: &Graph, start: usize, goal: usize) -> Option<(i32, Vec<usize>)> {
    let n = graph.adjacency_matrix.len();
    let mut dist = vec![i32::MAX; n];
    let mut parent = vec![None; n];
    dist[start] = 0;

    let mut pq = BinaryHeap::new();
    pq.push(State { cost: 0, node: start });

    while let Some(State { cost, node }) = pq.pop() {
        if node == goal {
            // Reconstruct path
            let mut path = Vec::new();
            let mut current = goal;
            while let Some(p) = parent[current] {
                path.push(current);
                current = p;
            }
            path.push(start);
            path.reverse();
            return Some((cost, path));
        }

        if cost > dist[node] {
            continue;
        }

        // Explore neighbors
        for (next, &edge_cost) in graph.adjacency_matrix[node].iter().enumerate() {
            if edge_cost > 0 { // Edge exists
                let next_cost = cost + edge_cost;
                if next_cost < dist[next] {
                    dist[next] = next_cost;
                    parent[next] = Some(node);
                    let f_score = next_cost + heuristic(graph, next, goal);
                    pq.push(State { cost: f_score, node: next });
                }
            }
        }
    }

    None // No path found
}

fn main() -> io::Result<()> {
    let graph_file = "/Users/saimouryab/Documents/DPS final/input_10000.txt";
    let graph = read_graph_from_file(graph_file)?;

    let start = 0; // Start node
    let goal = graph.adjacency_matrix.len() - 1; // Last node as goal

    let start_time = std::time::Instant::now();
    let result = a_star_serial(&graph, start, goal);
    let duration = start_time.elapsed();

    match result {
        Some((cost, path)) => {
            println!("Path found: {:?} with cost {}", path, cost);
        }
        None => {
            println!("No path found.");
        }
    }
    println!("Execution time: {:?}", duration);

    Ok(())
}
