use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicI32, Ordering};
use std::collections::BinaryHeap;
use std::thread;

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

    // Read the adjacency matrix size
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

fn dijkstra_parallel(graph: Arc<Graph>, start: usize, goal: usize, num_threads: usize) -> Option<(i32, Vec<usize>)> {
    let n = graph.adjacency_matrix.len();
    let distances = Arc::new((0..n).map(|_| AtomicI32::new(i32::MAX)).collect::<Vec<_>>());
    let parents = Arc::new(Mutex::new(vec![None; n]));
    let global_best_cost = Arc::new(AtomicI32::new(i32::MAX));
    let work_queues: Vec<Arc<Mutex<BinaryHeap<State>>>> = (0..num_threads).map(|_| Arc::new(Mutex::new(BinaryHeap::new()))).collect();

    distances[start].store(0, Ordering::Relaxed);
    work_queues[0].lock().unwrap().push(State { cost: 0, position: start });

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let graph = Arc::clone(&graph);
        let distances = Arc::clone(&distances);
        let parents = Arc::clone(&parents);
        let global_best_cost = Arc::clone(&global_best_cost);
        let work_queues = work_queues.clone();

        let handle = thread::spawn(move || {
            let mut local_queue = BinaryHeap::new();

            loop {
                if local_queue.is_empty() {
                    for queue in &work_queues {
                        let mut q = queue.lock().unwrap();
                        if !q.is_empty() {
                            local_queue.extend(q.drain());
                        }
                    }
                    if local_queue.is_empty() {
                        break;
                    }
                }

                if let Some(State { cost, position }) = local_queue.pop() {
                    if cost >= global_best_cost.load(Ordering::Relaxed) {
                        continue;
                    }

                    for neighbor in 0..graph.adjacency_matrix.len() {
                        if graph.adjacency_matrix[position][neighbor] == 1 {
                            let new_cost = cost + graph.weight_matrix[position][neighbor];
                            if new_cost < distances[neighbor].load(Ordering::Relaxed) {
                                distances[neighbor].store(new_cost, Ordering::Relaxed);
                                let mut parent_lock = parents.lock().unwrap();
                                parent_lock[neighbor] = Some(position);

                                let mut q = work_queues[thread_id].lock().unwrap();
                                q.push(State { cost: new_cost, position: neighbor });
                            }
                        }
                    }
                }
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_cost = distances[goal].load(Ordering::Relaxed);
    if final_cost == i32::MAX {
        return None;
    }

    let mut path = vec![];
    let mut current = goal;
    while let Some(parent) = parents.lock().unwrap()[current] {
        path.push(current);
        current = parent;
    }
    path.push(start);
    path.reverse();

    Some((final_cost, path))
}

fn main() {
    let filepath = "/Users/saimouryab/Documents/DPS final/input_10000.txt";
    let graph = Arc::new(read_graph_from_file(filepath).expect("Failed to read graph"));

    let start = 0;
    let goal = 99;
    let num_threads = 8; // number of threads
    let start_time = std::time::Instant::now();

    match dijkstra_parallel(graph, start, goal, num_threads) {
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


//"/Users/saimouryab/Documents/DPS final/input_46.txt";