use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::collections::BinaryHeap;

// Graph structure
#[derive(Debug)]
struct Graph {
    adjacency_matrix: Vec<Vec<i32>>,
    weight_matrix: Vec<Vec<i32>>,
}

#[derive(Debug, Eq, PartialEq)]
struct State {
    cost: i32,
    position: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cost.cmp(&self.cost) 
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn read_graph_from_file(filepath: &str) -> io::Result<Graph> {
    let file = File::open(filepath)?;
    let mut lines = BufReader::new(file).lines();

    // Read the first line (nodes and edges, but we only care about nodes here)
    let first_line = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing size line"))?;
    let n: usize = first_line?
        .split_whitespace()
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Invalid size line"))?
        .parse()
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid size line: {}", e)))?;

    // Read adjacency matrix
    let mut adjacency_matrix = vec![vec![0; n]; n];
    for i in 0..n {
        let line = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing adjacency matrix line"))?;
        adjacency_matrix[i] = line?
            .split_whitespace()
            .map(|x| x.parse::<i32>().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid adjacency value: {}", e))))
            .collect::<Result<Vec<_>, _>>()?;
    }

    // Read weight matrix
    let mut weight_matrix = vec![vec![0; n]; n];
    for i in 0..n {
        let line = lines
            .next()
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "Missing weight matrix line"))?;
        weight_matrix[i] = line?
            .split_whitespace()
            .map(|x| x.parse::<i32>().map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("Invalid weight value: {}", e))))
            .collect::<Result<Vec<_>, _>>()?;
    }

    Ok(Graph {
        adjacency_matrix,
        weight_matrix,
    })
}

// Dijkstra's Algorithm (Serial)
fn dijkstra_serial(graph: &Graph, start: usize, goal: usize) -> Option<(i32, Vec<usize>)> {
    let n = graph.adjacency_matrix.len();
    let mut distances = vec![i32::MAX; n];
    let mut previous = vec![None; n];
    let mut heap = BinaryHeap::new();

    // Initialize the starting node
    distances[start] = 0;
    heap.push(State {
        cost: 0,
        position: start,
    });

    while let Some(State { cost, position }) = heap.pop() {
        // Stop if we've reached the goal
        if position == goal {
            let mut path = vec![];
            let mut current = Some(goal);
            while let Some(node) = current {
                path.push(node);
                current = previous[node];
            }
            path.reverse();
            return Some((cost, path));
        }

        // Skip if this cost is not better
        if cost > distances[position] {
            continue;
        }

        // Update neighbors
        for neighbor in 0..n {
            if graph.adjacency_matrix[position][neighbor] == 1 {
                let next_cost = cost + graph.weight_matrix[position][neighbor];
                if next_cost < distances[neighbor] {
                    distances[neighbor] = next_cost;
                    previous[neighbor] = Some(position);
                    heap.push(State {
                        cost: next_cost,
                        position: neighbor,
                    });
                }
            }
        }
    }

    None 
}

fn main() {
    let filepath = "/Users/saimouryab/Documents/DPS final/input_10000.txt"; 
    let graph = read_graph_from_file(filepath).expect("Failed to read graph");

    let start = 0; // Starting node
    let goal = 99;  // Goal node
    let start_time = std::time::Instant::now();

    match dijkstra_serial(&graph, start, goal) {
        Some((cost, path)) => {
            println!("Shortest path: {:?} with cost: {}", path, cost);
        }
        None => {
            println!("No path found from node {} to node {}", start, goal);
        }
    }

    let duration = start_time.elapsed();
    println!("Execution time: {:?}", duration);
}


//"/Users/saimouryab/Documents/DPS final/input_46.txt"