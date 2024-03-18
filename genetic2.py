import random
import os
import time
import pandas as pd


#################################################################   Variables    #############################################################
# Colors to be used for display fo the diffent messages
Red = "\033[31m"  # Red text
Green = "\033[32m"  # Green text
BLUE = "\033[94m" # Blue text 
White = "\033[0m"  # Reset to default text color
Yellow = "\033[33m"  # Yellow text
approach = 'random'
cross_over_approach = 'random'
mode = 'simple'
tournament_size = 5 # default tournament size
# Constants
subfolder_name = 'grids'
ALPHABET = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
LETTERS = [] # converting the word entered by the user to a list of letters
processing_results = []
GRID_SIZE = 4
Population_Size = 100
MAX_GENERATIONS = 1000
CROSSOVER_RATE = 0.8
MUTATION_RATE = 0.2
grid_counter = 0
word_counter = 0
SUBGRID_SIZE = 2
original_grid = []
generation_info = []
columns = ['Word',  'Selection Algo.', 'Crossover Algo.' ,'Initial Grid','Solution', 'Word on Edge', 'Number of Solutions', 'Number Edges', 'Processing Time'] # columns for the results table
results_df = pd.DataFrame(columns=columns)
solution_columns = ['Word',  'Selection Algo.', 'Crossover Algo.' ,'Initial Grid','Solution',
                     'Word on Edge', 'Generations', 'Processing Time', 'Parent Selection Input', 'Parent Selection Output',
                     'Crossover Input','Crossover Output','Populations Created' ] # columns for the results table
solutions_df = pd.DataFrame(columns=solution_columns)
empirical_analysis = pd.DataFrame(columns=['Word', 'Selection Algorithm', 'Cross Over Algorithms', 'Execution Time', 'Number of Generations', 'Initial Grid', 'Solution'])

#################################################################   END Variables    #############################################################

################################################## Words that can be used to generate grid and check for edges#############################################################
words = [
    "cost", "crew", "crop", "dark", "data", "date", "dawn", "days", "dead", "deal", 
    "dean", "dear", "debt", "deep", "deny", "desk", "dial", "dick", "diet", "disc", 
    "disk", "does", "done", "door", "dose", "down", "draw", "drew", "drop", "drug", 
    "dual", "duke", "dust", "duty", "each", "earn", "ease", "east", "easy", "edge", 
    "else", "even", "ever", "evil", "exit", "face", "fact", "fail", "fair", "fall", 
    "farm", "fast", "fate", "fear", "feed", "feel", "feet", "fell", "felt", "file", 
    "fill", "film", "find", "fine", "fire", "firm", "fish", "five", "flat", "flow", 
    "food", "foot", "ford", "form", "fort", "four", "free", "from", "fuel", "full", 
    "fund", "gain", "game", "gate", "gave", "gear", "gene", "gift", "girl", "give", 
    "glad", "goal", "goes", "gold", "Golf", "gone", "good", "gray", "grew", "grey", 
    "grow", "gulf", "hair", "half", "hall", "hand", "hang", "hard", "harm", "hate", 
    "have", "head", "hear", "heat", "held", "hell", "help", "here", "hero", "high", 
    "hill", "hire", "hold", "hole", "holy", "home", "hope", "host", "hour", "huge", 
    "hung", "hunt", "hurt", "idea", "inch", "into", "iron", "item", "jack", "jane", 
    "jean", "john", "join", "jump", "jury", "just", "keen", "keep", "kent", "kept", 
    "kick", "kill", "kind", "king", "knee", "knew", "know", "lack", "lady", "laid", 
    "lake", "land", "lane", "last", "late", "lead", "left", "less", "life", "lift", 
    "like", "line", "link", "list", "live", "load", "loan", "lock", "logo", "long", 
    "look", "lord", "lose", "loss", "lost", "love", "luck", "made", "mail", "main", 
    "make", "male", "many", "Mark", "mass", "matt", "meal", "mean", "meat", "meet", 
    "menu", "mere", "mike", "mile", "milk", "mill", "mind", "mine", "miss", "mode", 
    "mood", "moon", "more", "most", "move", "much", "must", "name", "navy", "near", 
    "neck", "need", "news", "next", "nice", "nick", "nine", "none", "nose", "note", 
    "okay", "once", "only", "onto", "open", "oral", "over", "pace", "pack", "page"
]
#################################################################  End of Helper Functions    #############################################################


#################################################################   Genetic Algorithm Code    #############################################################
# function to generate a chromosome based on the initial grid, the function will fill the empty cells with letters from the alphabet
def generate_chromosome(initial_grid):
    flattened_grid = [cell for row in initial_grid for cell in row]
    existing_letters = set(cell for cell in flattened_grid if cell != '_')
    additional_letters_needed = GRID_SIZE - len(existing_letters)
    if additional_letters_needed > 0:
        additional_letters = random.sample([l for l in ALPHABET if l not in existing_letters], additional_letters_needed)
    else:
        additional_letters = []

    all_letters = list(existing_letters) + additional_letters # Combine existing and additional letters
    all_letters = [letter for letter in all_letters if letter.strip() != ''  and letter != ' '] # rmoving any whitespace
    flattened_grid = [l for l in flattened_grid if l.strip() != ''  and l != ' '] # am doing something wrong that is adding a whitepsace
    # Try to fill each empty cell with a letter from all_letters
    # added one support function to check if a letter can be placed in a given position
    for i, cell in enumerate(flattened_grid):
        if cell == '_':
            random.shuffle(all_letters)  # Shuffle to randomize the letter choice
            for letter in all_letters:
                if is_letter_replaceable(flattened_grid, letter, i):
                    flattened_grid[i] = letter
                    break  
    for i, cell in enumerate(flattened_grid):
            if cell == '_':
                for letter in all_letters:
                    if is_letter_replaceable(flattened_grid, letter, i):
                        flattened_grid[i] = letter
                        break  # Move to the next cell after placing a letter           
    return [flattened_grid[i:i + GRID_SIZE] for i in range(0, GRID_SIZE**2, GRID_SIZE)]

# function to compute the fitness of a chromosome based on the number of duplicates in the rows, 
# columns and subgrids starting with 12 and penalizing each duplicate
def compute_fitness(chromosome): 
    fitness = GRID_SIZE * 3  # intial fitness value is maximuim score, I will penalize each duplicate on the row, column or subgrid level.
    grid = chromosome
    
    # Check for unique letters in rows and subgrids
    for i in range(GRID_SIZE):
        row = grid[i]
        if len(set(row)) < GRID_SIZE:
            fitness -= 1  # Penalize for duplicate letter in row
        
        # Subgrid check, like this it is taking less time since it is in the same loop
        if i % 2 == 0:
            for j in [0, 2]:
                subgrid = grid[i][j:j+2] + grid[i+1][j:j+2]
                if len(set(subgrid)) < GRID_SIZE:
                    fitness -= 1  # less than 4 meaning there are duplicatd letters in the subgrid

    # Check for unique letters in columns and ensure all letters are present
    for j in range(GRID_SIZE):
        column = [grid[i][j] for i in range(GRID_SIZE)]
        if len(set(column)) < GRID_SIZE or not all(letter in column for letter in LETTERS):
            fitness -= 1  # Penalize for duplicates or missing letters in column

    return fitness

# helper function to check if a word exists on any of the grid edges 
# TODO: Granted am generating random grids, I will adjust this function once the instructor confirms how to know which word to search for.
def word_on_edge(chromosome, word): # TODO: remove the X parameter
    flattened_chromosome = [cell for row in chromosome for cell in row]
    edges = (
        flattened_chromosome[:4] +
        flattened_chromosome[12:] +
        [flattened_chromosome[i] for i in range(0, 16, 4)] +
        [flattened_chromosome[i] for i in range(3, 16, 4)]
    )
    edges_str = ''.join(edges[:4]) + ''.join(edges[12:]) + ''.join(edges[4:8]) + ''.join(edges[8:12])
    return word in edges_str or word[::-1] in edges_str

# function to select parents from the population based on random selection
# TODO: implement a selection function that selects the parents based on the fitness of the chromosomes
def select_parents(population, approach='random'):
    #print('len(population)', len(population))
    if(len(population) < 2):
        return population
    if approach == 'random':
        return random.sample(population, 2)
    if approach == 'wheel':
        fitnesses = [compute_fitness(chromosome) for chromosome in population]
        total_fitness = sum(fitnesses)
        selection_probs = [fitness / total_fitness for fitness in fitnesses]
        parent_indices = random.choices(range(len(population)), weights=selection_probs, k=2)   # Select two parents based on their selection probabilities
        return [population[parent_indices[0]], population[parent_indices[1]]]
    if approach == 'rank':
        fitnesses = [compute_fitness(chromosome) for chromosome in population]
        ranked_population = [chromosome for _, chromosome in sorted(zip(fitnesses, population), reverse=True)]
        return ranked_population[:2]
    return random.sample(population, 2) # default to random selection


# function to perform crossover between two parents
def crossover(parent1, parent2, method='random'):
    global original_grid
    if random.random() >= CROSSOVER_RATE:
        return parent1, parent2

    fixed_positions = {(row_idx, col_idx) for row_idx, row in enumerate(original_grid)
                   for col_idx, val in enumerate(row) if val != '_'}
        
    offspring1, offspring2 = [row[:] for row in parent1], [row[:] for row in parent2]

    if method == 'random':
        point = random.randint(1, GRID_SIZE - 1)
        for row in range(GRID_SIZE):
            if row >= point:
                for col in range(GRID_SIZE):
                    if (row, col) not in fixed_positions:
                        offspring1[row][col], offspring2[row][col] = offspring2[row][col], offspring1[row][col]

    elif method == 'row_swapping':
        subgrid_row_start = random.choice([0, 2])  # Only start at row 0 or row 2
        for i in range(2):  # Only two rows in each 2x2 subgrid for a 4x4 grid
            row = subgrid_row_start + i
            if row < GRID_SIZE and all((row, col) not in fixed_positions for col in range(GRID_SIZE)):
                offspring1[row], offspring2[row] = parent2[row][:], parent1[row][:]

    elif method == 'order_based':
        start, end = sorted(random.sample(range(GRID_SIZE), 2))
        for row in range(GRID_SIZE):
            if all((row, col) not in fixed_positions for col in range(start, end + 1)):
                temp = offspring1[row][start:end+1]
                offspring1[row][start:end+1] = offspring2[row][start:end+1]
                offspring2[row][start:end+1] = temp

    else:
        raise ValueError("Unknown crossover method specified.")

    return offspring1, offspring2


# function to perform mutation on a chromosome
def mutate(chromosome, initial_grid):
    if random.random() < MUTATION_RATE:
        flattened_grid = [cell for row in initial_grid for cell in row]
        flattened_chromosome = [cell for row in chromosome for cell in row]
        fixed_positions = {i for i, cell in enumerate(flattened_grid) if cell != '_'}
        possible_indices = [i for i in range(len(flattened_chromosome)) if i not in fixed_positions]
        if len(possible_indices) > 1:
            point1, point2 = random.sample(possible_indices, 2)
            flattened_chromosome[point1], flattened_chromosome[point2] = flattened_chromosome[point2], flattened_chromosome[point1]

        mutated_chromosome = [flattened_chromosome[i:i + GRID_SIZE] for i in range(0, GRID_SIZE**2, GRID_SIZE)]
        return mutated_chromosome

    return chromosome # No mutation performed
#################################################################   End of Genetic Algorithm Code    #############################################################


#################################################################  Helper Functions    #############################################################
# Helper function to add solution to dataframe
def add_solution(word, selection_algo, crossover_algo, initial_grid, solution, word_on_edge, generations, processing_time, parent_selection_input, parent_selection_output, crossover_input, crossover_output, populations_created):
    global solutions_df  # Reference the global DataFrame
    new_row = pd.DataFrame([{
        'Word': word,
        'Selection Algo.': selection_algo,
        'Crossover Algo.': crossover_algo,
        'Initial Grid': '\n'.join(' '.join(row) for row in initial_grid),
        'Solution': '\n'.join(' '.join(row) for row in solution),
        'Word on Edge': word_on_edge,
        'Generations': generations,
        'Processing Time': processing_time,
        'Parent Selection Input': str(parent_selection_input),
        'Parent Selection Output': str(parent_selection_output),
        'Crossover Input': str(crossover_input),
        'Crossover Output': str(crossover_output),
        'Populations Created': str(populations_created)
    }])
    #solutions_df = add_solution(selected_word, selection_approach, cross_over_approach, original_grid, individual, False, generation, solution_processing_time)

    # Append the new row to the DataFrame
    solutions_df = pd.concat([solutions_df, new_row], ignore_index=True)
    return solutions_df



# A helper function to add a new row to the results dataframe
def add_data(word,approach, cross_over_approach,  initial_grid, solution, word_on_edge, number_of_solutions, number_edge, processing_time):
    global results_df  # Reference the global DataFrame
    new_row = pd.DataFrame([{
        'Word': word,
        'Selection Algo.': approach,    
        'Crossover Algo.': cross_over_approach,
        'Initial Grid': [initial_grid],  # Wrap in a list to ensure it's treated as a single entry        
        'Solution': solution,
        'Word on Edge': word_on_edge,
        'Number of Solutions': number_of_solutions,
        'Number Edges': number_edge,
        'Processing Time': processing_time
    }])
    
    # Use pandas.concat to add the new row
    results_df = pd.concat([results_df, new_row], ignore_index=True)

# helper function to check if a letter can be placed in a given position
def is_letter_replaceable(flattened_grid, letter, position):
        row, col = divmod(position, GRID_SIZE) # Get the row and column indices from the position index
        row_start, col_start = row * GRID_SIZE, col # Get the start of the row and column
        if any(flattened_grid[row_start + position] == letter for position in range(GRID_SIZE)) : # Check if the letter is already in the row
            return False
        if any(flattened_grid[col_start + position * GRID_SIZE] == letter for position in range(GRID_SIZE)): # Check if the letter is already in the column
            return False
       
        return True

def print_grid(grid, word = ''):
    print(grid)
    if(word == ''):
        for row in grid: 
            print(' '.join(row))  
        print()
    else: #  the user passed a valid word so I will try to highlight it
        top_edge = ''.join(grid[0])
        bottom_edge = ''.join(grid[-1])
        left_edge = ''.join(row[0] for row in grid)
        right_edge = ''.join(row[-1] for row in grid)

        # Combine edges and check if the word is in any of them
        edges = top_edge + bottom_edge + left_edge + right_edge
        word_in_edge = word in edges

        for i, row in enumerate(grid):
            for j, cell in enumerate(row):
                if word_in_edge:
                    if (i == 0 and word in top_edge) or \
                    (i == len(grid) - 1 and word in bottom_edge) or \
                    (j == 0 and word in left_edge) or \
                    (j == len(row) - 1 and word in right_edge):
                        print(BLUE + cell + White, end=' ')
                    else:
                        print(cell, end=' ')
                else:
                    print(cell, end=' ')
            print()  # Newline after each row
        print()  # Extra newline for spacing

def validate_grid(grid):
    if len(grid) != 4:
        print_error("Error: The grid does not have 4 rows.") # The grid must contain 4 rows
        return False

    all_letters = set()
    for row in grid:
        all_letters.update(cell for cell in row if cell.strip())  # Ignore empty cells

    if len(all_letters) < 4:
        print_error("Error: The grid does not contain at least 4 unique letters.")
        return False

    #print_success("The grid is valid.")
    return True

def print_error(message):
    print(Red + message + White)

def print_success(message):
    print(Green + message + White)

def print_info(message):
    print(Yellow + message + White)


def generate_grid(word):
    global word_counter 
    if(word == ''): # if the user did not enter a word then I will generate a random word
        selected_word = random.choice(words)
    else:
        selected_word = word
    grid = [['_' for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
    
    for letter in selected_word:
        placed = False
        while not placed:
            row, col = random.randint(0, GRID_SIZE-1), random.randint(0, GRID_SIZE-1)
            # Check if the cell is '_' (empty) before placing a letter
            if grid[row][col] == '_':
                grid[row][col] = letter
                placed = True
                
    return grid, selected_word

def save_grid_to_file(grid, file_path):
    with open(file_path, 'w') as file:
        for row in grid:
            # Convert each cell to a string, replacing None with a space or another placeholder
            row_str = ' '.join(cell if cell is not None else "_" for cell in row)
            file.write(row_str + '\n')
def generate_and_save_grids( number_of_grids = 1):
    grids_folder = "grids"  # Folder to save grids, I could have prompted th user to enter the folder name
    os.makedirs(grids_folder, exist_ok=True)  # Create the folder if it doesn't exist.. Meh
    
    for i in range(1, number_of_grids):  # TODO: allow the user to specify the number of grids to generate
        grid, name = generate_grid('')
        file_name = f"{name}_{i}.txt"
        file_path = os.path.join(grids_folder, file_name)
        save_grid_to_file(grid, file_path)
        #print_success(f"Saved: {file_path}")
def read_grid_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            grid = []
            for line in file:
                # Use list to convert each character in the line into separate list elements
                row = list(line.strip())
                grid.append(row)
            return grid
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
#################################################################  End of Helper Functions    #############################################################

def process_grids():
    # Get the full path to the subfolder
    global grid_counter, results_df,selected_word, generation_info
    grid_counter = 0
    script_dir = os.path.dirname(os.path.abspath(__file__))
    subfolder_path = os.path.join(script_dir, subfolder_name)
    try:
        files = [file for file in os.listdir(subfolder_path) if file.endswith('.txt')]
    except FileNotFoundError:
        print(f"Subfolder '{subfolder_name}' not found.")
        return
    
    if not files:
        print("No .txt files found in the specified subfolder.")
        return

    # Process each file in the subfolder
    for file_name in files:
        file_path = os.path.join(subfolder_path, file_name)
        process_file(file_path)
        selected_word =  os.path.basename(file_path).split('_')[0] # on my mac this works, ensure it works on windows
        generation_df = pd.DataFrame(generation_info)
        csv_file_path = selected_word+"_generation_info.csv"  # Save the DataFrame to a CSV file
        generation_df.to_csv(csv_file_path, index=False) # i made my own index,.. not sure why
        print(f"Saved generation information to {csv_file_path}")
    for result in processing_results:
        print(f"{grid_counter}. File: {result['file_name']}, Status: {result['status']}, Time: {result['processing_time']:.2f} seconds")
    print(f"Processed {len(processing_results)} files.")

def process_file(file_path):
    global grid_counter, results_df, original_grid
    selected_word =  os.path.basename(file_path).split('_')[0] # on my mac this works, ensure it works on windows
    _index = 0
    grid = read_grid_from_file(file_path)
    process_grid(grid, selected_word, approach, cross_over_approach)


def process_grid(grid, selected_word, selection_approach, cross_over_approach, empirical = False):
    global grid_counter, results_df, original_grid, generation_info, empirical_analysis, solutions_df
    generation_info = []
    _index = 0
    print(f"Processing grid with word { Red + selected_word  + White} using selection approach: {selection_approach}")
    found_solutions = []
    edge_solutions = []
    found_first_solution = False
    word_on_edge_flag = False
    start_time = time.time()  # Start timer
    solution_found = False # used for the empirical analysis
    solution = None # used for the empirical analysis
    execution_time = None # used for the empirical analysis
    if validate_grid(grid) == False: # If the grid is invalid, skip to the next file
        print_error(f"Skipping grid {grid_counter} due to invalid grid.")
        return None
    original_grid = grid    
    original_grid = [[element for element in sublist if element not in [ ' ']] for sublist in original_grid]
    population = [generate_chromosome(grid) for _ in range(Population_Size)]
    population = [nested_list for nested_list in population if not any('_' in sublist for sublist in nested_list)] #added this to invalidate any chromosome that the algorithm was not able to solve
    if(len(population)==0):  # incase all generated populations are invalid
        return None
    # sort the initial population based on the fitness of the chromosomes
    population = sorted(population, key=compute_fitness, reverse=True)
    gen_prev = 0
    
    parent_selection_input = None
    parent_selection_output = None
    crossover_input = None
    crossover_output = None
    populations_created = []

    for generation in range(MAX_GENERATIONS): # Loop through the generations to find all possible solutions
        solution_processing_time = None # Reset the solution processing time for each generation
        solution_start_time = time.time()  # Start timer for this generation
        populations_created.append(list(population))  # populate the inital list of populations
        new_population = []
        while len(new_population) < Population_Size:
            parent_selection_input = list(population)
            parents = select_parents(population, approach=selection_approach)
            parent_selection_output = parents
            crossover_input = (parents[0], parents[1])
            child1, child2 = crossover(parents[0], parents[1], method=cross_over_approach)
            crossover_output = (child1, child2)
            mutate(child1, grid)
            mutate(child2, grid)
            new_population.extend([child1, child2])
            
        population = sorted(new_population, key=compute_fitness, reverse=True)
        if(empirical == True):
            for individual in population:
                if compute_fitness(individual) == GRID_SIZE * 3: 
                    solution = individual
                    solution_found = True
                    execution_time = time.time() - solution_start_time
                    sol = None
                    new_row=None
                    if( solution_found):
                        sol = '\n'.join([''.join(row) for row in solution])
                    else:
                        sol = "Not Found"
                    new_row = pd.DataFrame([{ # add the empirical results to the dataframe
                                                'Word': selected_word,
                                                'Selection Algorithm': selection_approach,
                                                'Cross Over Algorithms': cross_over_approach,
                                                'Execution Time': execution_time,
                                                'Number of Generations': generation + 1 if solution_found else MAX_GENERATIONS,
                                                'Initial Grid': '\n'.join([''.join(row) for row in original_grid]),
                                                'Solution': sol
                                            }])
                    empirical_analysis = pd.concat([empirical_analysis, new_row], ignore_index=True)
                    

            
            # new_row = pd.DataFrame([{ # add the empirical results to the dataframe
            #     'Word': selected_word,
            #     'Selection Algorithm': selection_approach,
            #     'Cross Over Algorithms': cross_over_approach,
            #     'Execution Time': execution_time,
            #     'Number of Generations': generation + 1 if solution_found else MAX_GENERATIONS,
            #     'Initial Grid': '\n'.join([''.join(row) for row in original_grid]),
            #     'Solution': sol
            # }])
            # empirical_analysis = pd.concat([empirical_analysis, new_row], ignore_index=True)
            #if solution_found:
                
                #break

            
        ######### end if empirical
        else: # none empirical
            #print('Generation:', generation, 'Fitness:', compute_fitness(population[0]))    
            # checkinf if one or more solutions are found
            for individual in population:
                fitness = compute_fitness(individual)
                if compute_fitness(individual) == GRID_SIZE * 3 and individual not in found_solutions and individual not in edge_solutions:
                    if word_on_edge(individual, selected_word):
                        word_on_edge_flag = True
                        edge_solutions.append(individual)
                        print_success('Word found on edge:')
                        print_grid(individual, selected_word)
                        solution_processing_time = time.time() - solution_start_time
                        solutions_df = add_solution(selected_word, selection_approach, cross_over_approach, original_grid, individual, True, generation - gen_prev, solution_processing_time, parent_selection_input, parent_selection_output, crossover_input, crossover_output, populations_created)
                        gen_prev = generation
                        print()
                    else:
                        print_success('Word found:')
                        found_solutions.append(individual)
                        solution_processing_time = time.time() - solution_start_time
                        solutions_df = add_solution(selected_word, selection_approach, cross_over_approach, original_grid, individual, False, generation - gen_prev, solution_processing_time, parent_selection_input, parent_selection_output, crossover_input, crossover_output, populations_created)
                        gen_prev = generation
                        print_grid(individual, '')
                        print()
                for individual in population:
                    
                    generation_info.append({
                        'Index': _index,
                        'Word': selected_word,
                        'Original Grid': '\n'.join([''.join(row) for row in original_grid]),
                        'Generation': generation,
                        'Individual': '\n'.join([''.join(row) for row in individual]),
                        'Fitness': compute_fitness(individual),
                        'Parent Swap Approach': selection_approach
                    })
                    _index += 1
            #print(f"Generation {generation} completed. Found {len(found_solutions)} solutions.")
            processing_time = time.time() - start_time
            add_data(selected_word,  selection_approach, cross_over_approach, grid, [], word_on_edge_flag, len(found_solutions), len(edge_solutions), processing_time)
    #if(empirical == True):
            

    grid_counter += 1

def validate_word(word):
    return len(word) == 4 and word.isalpha() and word.isupper() and len(set(word)) == len(word)

# this helper function will prompt the user to enter a valid word
def get_valid_word(prompt):
    while True:
        selected_word = input(prompt).strip().upper()
        if validate_word(selected_word):
            return selected_word
        else:
            print_error("Invalid input. Please enter a 4-letter word with no repeated letters.")

def read_user_input(question, valid_words):
    while True:
        user_input = input(question).strip().lower()
        if user_input in valid_words:
            return user_input
        else:
            print(f"Invalid input. Please enter one of the following words: {', '.join(valid_words)}")


if __name__ == "__main__":
    #global selected_word, approach, cross_over_approach
    grid_counter = 0
    mode = read_user_input(f'Enter word { BLUE }simple {White} for a simple grid solver or { Red}performance {White} for interactive aproach and {Red} empirical {White } for anaysis mode: ', ['simple', 'performance', 'empirical'])
    if(mode == 'empirical'):
        print_info('Empirical mode selected, this mode will try all possible combinations of words and selection and crossover algorithms')
        print()
        selected_word = get_valid_word("Enter a 4 letter word with no repated letters: ")
        grid, w = generate_grid(selected_word)
        print_info(f"Generated grid for word '{selected_word}':")
        print_grid(grid)
        print('************************** Chromosome RANDOM ****************************************')
        process_grid(grid, selected_word, approach, 'random', True)
        ########################################
        print('************************** Chromosome ROW SWAPPING ****************************************')
        process_grid(grid, selected_word, approach, 'row_swapping', True)
        print('************************** Chromosome Order Based ****************************************')
        process_grid(grid, selected_word, approach, 'order_based', True)
        print('************************** Parent RANDOM ****************************************')
        process_grid(grid, selected_word, 'random', 'random', True)
        print('************************** Parent Selection Wheel ****************************************')
        process_grid(grid, selected_word, 'wheel' , 'random', True)
        print('************************** Parent Selection Rank ****************************************')
        process_grid(grid, selected_word, 'rank', 'random', True)
        selected_word = 'wear'
        grid, w = generate_grid(selected_word)
        print('************************** Chromosome RANDOM ****************************************')
        process_grid(grid, selected_word, approach, 'random', True)
        print('************************** Chromosome ROW SWAPPING ****************************************')
        process_grid(grid, selected_word, approach, 'row_swapping', True)
        print('************************** Chromosome Order Based ****************************************')
        process_grid(grid, selected_word, approach, 'order_based', True)

        print('************************** Parent RANDOM ****************************************')
        process_grid(grid, selected_word, 'random', 'random', True)
        print('************************** Parent Selection Wheel ****************************************')
        process_grid(grid, selected_word, 'wheel' , 'random', True)
        print('************************** Parent Selection Rank ****************************************')
        process_grid(grid, selected_word, 'rank', 'random', True)
        selected_word = 'ford'
        grid, w = generate_grid(selected_word)
        print('************************** Chromosome RANDOM ****************************************')
        process_grid(grid, selected_word, approach, 'random', True)
        print('************************** Chromosome ROW SWAPPING ****************************************')
        process_grid(grid, selected_word, approach, 'row_swapping', True)
        print('************************** Chromosome Order Based ****************************************')
        process_grid(grid, selected_word, approach, 'order_based', True)

        print('************************** Parent RANDOM ****************************************')
        process_grid(grid, selected_word, 'random', 'random', True)
        print('************************** Parent Selection Wheel ****************************************')
        process_grid(grid, selected_word, 'wheel' , 'random', True)
        print('************************** Parent Selection Rank ****************************************')
        process_grid(grid, selected_word, 'rank', 'random', True)
        ######
        selected_word = 'bord'
        grid, w = generate_grid(selected_word)


        print('************************** Chromosome RANDOM ****************************************')
        process_grid(grid, selected_word, approach, 'random', True)
        print('************************** Chromosome ROW SWAPPING ****************************************')
        process_grid(grid, selected_word, approach, 'row_swapping', True)
        print('************************** Chromosome Order Based ****************************************')
        process_grid(grid, selected_word, approach, 'order_based', True)

        print('************************** Parent RANDOM ****************************************')
        process_grid(grid, selected_word, 'random', 'random', True)
        print('************************** Parent Selection Wheel ****************************************')
        process_grid(grid, selected_word, 'wheel' , 'random', True)
        print('************************** Parent Selection Rank ****************************************')
        process_grid(grid, selected_word, 'rank', 'random', True)

        selected_word = 'word'
        grid, w = generate_grid(selected_word)


        print('************************** Chromosome RANDOM ****************************************')
        process_grid(grid, selected_word, approach, 'random', True)
        print('************************** Chromosome ROW SWAPPING ****************************************')
        process_grid(grid, selected_word, approach, 'row_swapping', True)
        print('************************** Chromosome Order Based ****************************************')
        process_grid(grid, selected_word, approach, 'order_based', True)

        print('************************** Parent RANDOM ****************************************')
        process_grid(grid, selected_word, 'random', 'random', True)
        print('************************** Parent Selection Wheel ****************************************')
        process_grid(grid, selected_word, 'wheel' , 'random', True)
        print('************************** Parent Selection Rank ****************************************')
        process_grid(grid, selected_word, 'rank', 'random', True)

        selected_word = 'word'
        grid, w = generate_grid(selected_word)


        print('************************** Chromosome RANDOM ****************************************')
        process_grid(grid, selected_word, approach, 'random', True)
        print('************************** Chromosome ROW SWAPPING ****************************************')
        process_grid(grid, selected_word, approach, 'row_swapping', True)
        print('************************** Chromosome Order Based ****************************************')
        process_grid(grid, selected_word, approach, 'order_based', True)

        print('************************** Parent RANDOM ****************************************')
        process_grid(grid, selected_word, 'random', 'random', True)
        print('************************** Parent Selection Wheel ****************************************')
        process_grid(grid, selected_word, 'wheel' , 'random', True)
        print('************************** Parent Selection Rank ****************************************')
        process_grid(grid, selected_word, 'rank', 'random', True)

        ######
        generation_df = pd.DataFrame(generation_info)
        csv_file_path = selected_word+"_generation_info.csv"
        columns = ['Word',  'Selection Algo.', 'Crossover Algo.' ,'Initial Grid','Solution', 'Word on Edge', 'Number of Solutions', 'Number Edges', 'Processing Time'] # columns for the results table
        generation_df.to_csv(csv_file_path, index=False) # i made my own index,.. not sure why
        csv_file_path = selected_word+"_processing_info.csv"
        results_df.to_csv(csv_file_path, index=False)
        csv_file_path = selected_word+"_results.csv"
        solutions_df.to_csv(csv_file_path, index=False)
        csv_file_path = "empirical_results_2.csv"
        empirical_analysis.to_csv(csv_file_path, index=False)
    else: # for performance mode
        approach = read_user_input("Enter the selection approach to use (random', wheel, rank): ", ['random', 'wheel', 'rank'])
        cross_over_approach = read_user_input("Enter the cross over approach to use random, row_swapping, order_based: ", ['random', 'row_swapping', 'order_based'])
        number_of_grids = int(input("Enter the number of grids to generate: "))
        generate_and_save_grids(number_of_grids)
        process_grids()
    
    if not results_df.empty:
        print(results_df.to_string(max_rows=100, max_colwidth=50))
    else:
        print_error("No Solutions found")

    

