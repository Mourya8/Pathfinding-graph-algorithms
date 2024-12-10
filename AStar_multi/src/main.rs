use std::cmp::Ordering as CmpOrdering;
use std::collections::BinaryHeap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicI32, Ordering as AtomicOrdering};
use std::thread;

#[derive(Debug)]
struct Graph {
    adjacency_matrix: Vec<Vec<i32>>,
}

#[derive(Debug)]
struct State {
    cost: i32,
    node: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> CmpOrdering {
        other.cost.cmp(&self.cost) // Min-heap
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<CmpOrdering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for State {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost && self.node == other.node
    }
}

impl Eq for State {}

fn read_graph_from_file(filepath: &str) -> io::Result<Graph> {
    let file = File::open(filepath)?;
    let mut lines = BufReader::new(file).lines();

    // Read the number of nodes
    let first_line = lines.next().unwrap_or_else(|| {
        panic!("Failed to read the first line (number of nodes) from the file.")
    })?;
    let n: usize = first_line
        .split_whitespace()
        .next()
        .expect("Expected a number of nodes but got an empty line.")
        .parse()
        .expect("Failed to parse the number of nodes as a valid integer.");

    // Read adjacency matrix
    let mut adjacency_matrix = vec![vec![0; n]; n];
    for i in 0..n {
        let line = lines.next().unwrap_or_else(|| {
            panic!(
                "Expected line {} of the adjacency matrix, but the file ended prematurely.",
                i + 1
            )
        })?;
        adjacency_matrix[i] = line
            .split_whitespace()
            .map(|x| x.parse::<i32>().expect("Failed to parse an adjacency matrix entry as an integer."))
            .collect();
        if adjacency_matrix[i].len() != n {
            panic!(
                "Adjacency matrix row {} does not have the expected number of entries. Expected {}, found {}.",
                i + 1,
                n,
                adjacency_matrix[i].len()
            );
        }
    }

    Ok(Graph { adjacency_matrix })
}


fn heuristic(_: &Graph, _: usize, _: usize) -> i32 {
    0 
}

fn a_star_parallel(
    graph: Arc<Graph>,
    start: usize,
    goal: usize,
    num_threads: usize,
) -> Option<(i32, Vec<usize>)> {
    let n = graph.adjacency_matrix.len();
    let distances = Arc::new((0..n).map(|_| Arc::new(AtomicI32::new(i32::MAX))).collect::<Vec<_>>());
    let parents = Arc::new(Mutex::new(vec![None; n]));
    let global_best_cost = Arc::new(AtomicI32::new(i32::MAX));
    let best_path = Arc::new(Mutex::new(Vec::new()));
    let work_queues: Vec<Arc<Mutex<BinaryHeap<State>>>> = (0..num_threads)
        .map(|_| Arc::new(Mutex::new(BinaryHeap::new())))
        .collect();

    distances[start].store(0, AtomicOrdering::Relaxed);
    work_queues[0].lock().unwrap().push(State {
        cost: 0,
        node: start,
    });

    let mut handles = vec![];

    for thread_id in 0..num_threads {
        let graph = Arc::clone(&graph);
        let distances = Arc::clone(&distances);
        let parents = Arc::clone(&parents);
        let global_best_cost = Arc::clone(&global_best_cost);
        let best_path = Arc::clone(&best_path);
        let work_queues = work_queues.clone();

        let handle = thread::spawn(move || {
            let mut local_queue = BinaryHeap::new();

            loop {
                if local_queue.is_empty() {
                    for i in 0..num_threads {
                        let mut queue = work_queues[i].lock().unwrap();
                        if !queue.is_empty() {
                            local_queue.extend(queue.drain());
                            break;
                        }
                    }

                    if local_queue.is_empty() {
                        return;
                    }
                }

                if let Some(State { cost, node }) = local_queue.pop() {
                    if cost >= global_best_cost.load(AtomicOrdering::Relaxed) {
                        continue;
                    }

                    if node == goal {
                        let mut path = vec![];

                        let mut current = Some(node);
                        {
                            let parents_lock = parents.lock().unwrap();
                            while let Some(c) = current {
                                path.push(c);
                                current = parents_lock[c];
                            }
                        }
                        path.push(start);
                        path.reverse();

                        let mut path_lock = best_path.lock().unwrap();
                        if cost < global_best_cost.load(AtomicOrdering::Relaxed) {
                            *path_lock = path.clone();
                            global_best_cost.store(cost, AtomicOrdering::Relaxed);
                        }
                        return;
                    }

                    for (next, &edge_cost) in graph.adjacency_matrix[node].iter().enumerate() {
                        if edge_cost > 0 {
                            let next_cost = cost + edge_cost;
                            let current_cost = distances[next].load(AtomicOrdering::Relaxed);

                            if next_cost < current_cost {
                                distances[next].store(next_cost, AtomicOrdering::Relaxed);
                                parents.lock().unwrap()[next] = Some(node);

                                let f_score = next_cost;
                                local_queue.push(State { cost: f_score, node: next });

                                let mut queue = work_queues[thread_id].lock().unwrap();
                                queue.push(State { cost: f_score, node: next });
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

    let final_cost = global_best_cost.load(AtomicOrdering::Relaxed);
    if final_cost == i32::MAX {
        None
    } else {
        let path = best_path.lock().unwrap().clone();
        Some((final_cost, path))
    }
}

fn main() {
    let graph_file = "/Users/saimouryab/Documents/DPS final/input_10000.txt";
    let graph = Arc::new(read_graph_from_file(graph_file).expect("Failed to read graph"));

    let start = 0;
    let goal = graph.adjacency_matrix.len() - 1;
    let num_threads = 8;

    let start_time = std::time::Instant::now();
    let result = a_star_parallel(graph, start, goal, num_threads);
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
}




//Users/saimouryab/Documents/DPS final/input_100.txt

