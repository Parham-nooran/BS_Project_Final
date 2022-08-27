from __future__ import absolute_import
import math
import time
import random
import cv2
import matplotlib.pyplot as plt



class Solver(object):

    def __init__(self, maze, quiet_mode, neighbor_method):

        self.maze = maze
        self.neighbor_method = neighbor_method
        self.name = ""
        self.quiet_mode = quiet_mode

    def solve(self):
        raise NotImplementedError

    def get_name(self):
        raise self.name

    def get_path(self):
        return self.path


class BreadthFirst(Solver):

    def __init__(self, maze, quiet_mode=False, neighbor_method="fancy"):

        self.name = "Breadth First Recursive"
        super().__init__(maze, neighbor_method, quiet_mode)

    def solve(self):

        current_level = [self.maze.entry_coor]  # Stack of cells at current level of search
        path = list()  # To track path of solution cell coordinates

        print("\nSolving the maze with breadth-first search...")
        time_start = time.clock()

        while True:  # Loop until return statement is encountered
            next_level = list()

            while current_level:  # While still cells left to search on current level
                k_curr, l_curr = current_level.pop(0)  # Search one cell on the current level
                self.maze.grid[k_curr][l_curr].visited = True  # Mark current cell as visited
                path.append(((k_curr, l_curr), False))  # Append current cell to total search path

                if (k_curr, l_curr) == self.maze.exit_coor:  # Exit if current cell is exit cell
                    if not self.quiet_mode:
                        print("Number of moves performed: {}".format(len(path)))
                        print("Execution time for algorithm: {:.4f}".format(time.clock() - time_start))
                    return path

                neighbour_coors = self.maze.find_neighbours(k_curr, l_curr)  # Find neighbour indicies
                neighbour_coors = self.maze.validate_neighbours_solve(neighbour_coors, k_curr,
                                                                  l_curr, self.maze.exit_coor[0],
                                                                  self.maze.exit_coor[1], self.neighbor_method)

                if neighbour_coors is not None:
                    for coor in neighbour_coors:
                        next_level.append(coor)  # Add all existing real neighbours to next search level

            for cell in next_level:
                current_level.append(cell)  # Update current_level list with cells for nex search level


class BiDirectional(Solver):

    def __init__(self, maze, quiet_mode=False, neighbor_method="fancy"):

        super().__init__(maze, neighbor_method, quiet_mode)
        self.name = "Bi Directional"

    def solve(self):


        grid = self.maze.grid
        k_curr, l_curr = self.maze.entry_coor            
        p_curr, q_curr = self.maze.exit_coor             
        grid[k_curr][l_curr].visited = True    
        grid[p_curr][q_curr].visited = True    
        backtrack_kl = list()                  
        backtrack_pq = list()                  
        path_kl = list()                       
        path_pq = list()                       

        if not self.quiet_mode:
            print("\nSolving the maze with bidirectional depth-first search...")
        time_start = time.clock()

        while True:   
            neighbours_kl = self.maze.find_neighbours(k_curr, l_curr)    
            real_neighbours_kl = [neigh for neigh in neighbours_kl if not grid[k_curr][l_curr].is_walls_between(grid[neigh[0]][neigh[1]])]
            neighbours_kl = [neigh for neigh in real_neighbours_kl if not grid[neigh[0]][neigh[1]].visited]

            neighbours_pq = self.maze.find_neighbours(p_curr, q_curr)    
            real_neighbours_pq = [neigh for neigh in neighbours_pq if not grid[p_curr][q_curr].is_walls_between(grid[neigh[0]][neigh[1]])]
            neighbours_pq = [neigh for neigh in real_neighbours_pq if not grid[neigh[0]][neigh[1]].visited]

            if len(neighbours_kl) > 0: 
                backtrack_kl.append((k_curr, l_curr))             
                path_kl.append(((k_curr, l_curr), False))       
                k_next, l_next = random.choice(neighbours_kl)    
                grid[k_next][l_next].visited = True               
                k_curr = k_next
                l_curr = l_next

            elif len(backtrack_kl) > 0:              
                path_kl.append(((k_curr, l_curr), True))  
                k_curr, l_curr = backtrack_kl.pop()      

            if len(neighbours_pq) > 0:                       
                backtrack_pq.append((p_curr, q_curr))         
                path_pq.append(((p_curr, q_curr), False))      
                p_next, q_next = random.choice(neighbours_pq)   
                grid[p_next][q_next].visited = True             
                p_curr = p_next
                q_curr = q_next

            elif len(backtrack_pq) > 0:                  
                path_pq.append(((p_curr, q_curr), True))  
                p_curr, q_curr = backtrack_pq.pop()       

            if any((True for n_kl in real_neighbours_kl if (n_kl, False) in path_pq)):
                path_kl.append(((k_curr, l_curr), False))
                path = [p_el for p_tuple in zip(path_kl, path_pq) for p_el in p_tuple]  
                if not self.quiet_mode:
                    print("Number of moves performed: {}".format(len(path)))
                    print("Execution time for algorithm: {:.4f}".format(time.clock() - time_start))
                return path

            elif any((True for n_pq in real_neighbours_pq if (n_pq, False) in path_kl)):
                path_pq.append(((p_curr, q_curr), False))
                path = [p_el for p_tuple in zip(path_kl, path_pq) for p_el in p_tuple]  
                if not self.quiet_mode:
                    print("Number of moves performed: {}".format(len(path)))
                    print("Execution time for algorithm: {:.4f}".format(time.clock() - time_start))
                return path


class DepthFirstBacktracker(Solver):

    def __init__(self, maze, quiet_mode=False,  neighbor_method="fancy"):
        # logging.debug('Class DepthFirstBacktracker ctor called')

        super().__init__(maze, neighbor_method, quiet_mode)
        self.name = "Depth First Backtracker"

    def solve(self):
        # logging.debug("Class DepthFirstBacktracker solve called")
        k_curr, l_curr = self.maze.entry_coor      # Where to start searching
        self.maze.grid[k_curr][l_curr].visited = True     # Set initial cell to visited
        visited_cells = list()                  # Stack of visited cells for backtracking
        path = list()                           # To track path of solution and backtracking cells
        if not self.quiet_mode:
            print("\nSolving the maze with depth-first search...")

        time_start = time.time()

        while (k_curr, l_curr) != self.maze.exit_coor:     # While the exit cell has not been encountered
            neighbour_indices = self.maze.find_neighbours(k_curr, l_curr)    # Find neighbour indices
            neighbour_indices = self.maze.validate_neighbours_solve(neighbour_indices, k_curr,
                l_curr, self.maze.exit_coor[0], self.maze.exit_coor[1], self.neighbor_method)

            if neighbour_indices is not None:   # If there are unvisited neighbour cells
                visited_cells.append((k_curr, l_curr))              # Add current cell to stack
                path.append(((k_curr, l_curr), False))  # Add coordinates to part of search path
                k_next, l_next = random.choice(neighbour_indices)   # Choose random neighbour
                self.maze.grid[k_next][l_next].visited = True                 # Move to that neighbour
                k_curr = k_next
                l_curr = l_next

            elif len(visited_cells) > 0:              # If there are no unvisited neighbour cells
                path.append(((k_curr, l_curr), True))   # Add coordinates to part of search path
                k_curr, l_curr = visited_cells.pop()    # Pop previous visited cell (backtracking)

        path.append(((k_curr, l_curr), False))  # Append final location to path
        if not self.quiet_mode:
            print("Number of moves performed: {}".format(len(path)))
            print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

        return path

class Cell(object):
    def __init__(self, row, col):
        self.row = row
        self.col = col
        self.visited = False
        self.active = False
        self.is_entry_exit = None
        self.walls = {"top": True, "right": True, "bottom": True, "left": True}
        self.neighbours = list()

    def is_walls_between(self, neighbour):
        if self.row - neighbour.row == 1 and self.walls["top"] and neighbour.walls["bottom"]:
            return True
        elif self.row - neighbour.row == -1 and self.walls["bottom"] and neighbour.walls["top"]:
            return True
        elif self.col - neighbour.col == 1 and self.walls["left"] and neighbour.walls["right"]:
            return True
        elif self.col - neighbour.col == -1 and self.walls["right"] and neighbour.walls["left"]:
            return True

        return False

    def remove_walls(self, neighbour_row, neighbour_col):
        if self.row - neighbour_row == 1:
            self.walls["top"] = False
            return True, ""
        elif self.row - neighbour_row == -1:
            self.walls["bottom"] = False
            return True, ""
        elif self.col - neighbour_col == 1:
            self.walls["left"] = False
            return True, ""
        elif self.col - neighbour_col == -1:
            self.walls["right"] = False
            return True, ""
        return False

    def set_as_entry_exit(self, entry_exit, row_limit, col_limit):
        if self.row == 0:
            self.walls["top"] = False
        elif self.row == row_limit:
            self.walls["bottom"] = False
        elif self.col == 0:
            self.walls["left"] = False
        elif self.col == col_limit:
            self.walls["right"] = False

        self.is_entry_exit = entry_exit


algorithm_list = ["dfs_backtrack", "bin_tree"]

def depth_first_recursive_backtracker( maze, start_coor ):
        k_curr, l_curr = start_coor             # Where to start generating
        path = [(k_curr, l_curr)]               # To track path of solution
        maze.grid[k_curr][l_curr].visited = True     # Set initial cell to visited
        visit_counter = 1                       # To count number of visited cells
        visited_cells = list()                  # Stack of visited cells for backtracking

        print("\nGenerating the maze with depth-first search...")
        time_start = time.time()

        while visit_counter < maze.grid_size:     # While there are unvisited cells
            neighbour_indices = maze.find_neighbours(k_curr, l_curr)    # Find neighbour indicies
            neighbour_indices = maze._validate_neighbours_generate(neighbour_indices)

            if neighbour_indices is not None:   # If there are unvisited neighbour cells
                visited_cells.append((k_curr, l_curr))              # Add current cell to stack
                k_next, l_next = random.choice(neighbour_indices)     # Choose random neighbour
                maze.grid[k_curr][l_curr].remove_walls(k_next, l_next)   # Remove walls between neighbours
                maze.grid[k_next][l_next].remove_walls(k_curr, l_curr)   # Remove walls between neighbours
                maze.grid[k_next][l_next].visited = True                 # Move to that neighbour
                k_curr = k_next
                l_curr = l_next
                path.append((k_curr, l_curr))   # Add coordinates to part of generation path
                visit_counter += 1

            elif len(visited_cells) > 0:  # If there are no unvisited neighbour cells
                k_curr, l_curr = visited_cells.pop()      # Pop previous visited cell (backtracking)
                path.append((k_curr, l_curr))   # Add coordinates to part of generation path

        print("Number of moves performed: {}".format(len(path)))
        print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

        maze.grid[maze.entry_coor[0]][maze.entry_coor[1]].set_as_entry_exit("entry",
            maze.num_rows-1, maze.num_cols-1)
        maze.grid[maze.exit_coor[0]][maze.exit_coor[1]].set_as_entry_exit("exit",
            maze.num_rows-1, maze.num_cols-1)

        for i in range(maze.num_rows):
            for j in range(maze.num_cols):
                maze.grid[i][j].visited = False      # Set all cells to unvisited before returning grid

        maze.generation_path = path

def binary_tree( maze, start_coor ):
    time_start = time.time()

    for i in range(0, maze.num_rows):

        if( i == maze.num_rows - 1 ):
            for j in range(0, maze.num_cols-1):
                maze.grid[i][j].remove_walls(i, j+1)
                maze.grid[i][j+1].remove_walls(i, j)

            break

        for j in range(0, maze.num_cols):

            if( j == maze.num_cols-1 ):
                maze.grid[i][j].remove_walls(i+1, j)
                maze.grid[i+1][j].remove_walls(i, j)
                continue

            remove_top = random.choice([True,False])

            if remove_top:
                maze.grid[i][j].remove_walls(i+1, j)
                maze.grid[i+1][j].remove_walls(i, j)
            else:
                maze.grid[i][j].remove_walls(i, j+1)
                maze.grid[i][j+1].remove_walls(i, j)

    print("Number of moves performed: {}".format(maze.num_cols * maze.num_rows))
    print("Execution time for algorithm: {:.4f}".format(time.time() - time_start))

    maze.grid[maze.entry_coor[0]][maze.entry_coor[1]].set_as_entry_exit("entry",
        maze.num_rows-1, maze.num_cols-1)
    maze.grid[maze.exit_coor[0]][maze.exit_coor[1]].set_as_entry_exit("exit",
        maze.num_rows-1, maze.num_cols-1)

    path = list()
    visit_counter = 0
    visited = list()

    k_curr, l_curr = (maze.num_rows-1, maze.num_cols-1)
    path.append( (k_curr,l_curr) )

    begin_time = time.time()

    while visit_counter < maze.grid_size:     # While there are unvisited cells

        possible_neighbours = list()

        try:
            if not maze.grid[k_curr-1][l_curr].visited and k_curr != 0:
                if not maze.grid[k_curr][l_curr].is_walls_between(maze.grid[k_curr-1][l_curr]):
                    possible_neighbours.append( (k_curr-1,l_curr))
        except:
            print()

        try:
            if not maze.grid[k_curr][l_curr-1].visited and l_curr != 0:
                if not maze.grid[k_curr][l_curr].is_walls_between(maze.grid[k_curr][l_curr-1]):
                    possible_neighbours.append( (k_curr,l_curr-1))
        except:
            print()

        if len( possible_neighbours ) != 0:
            k_next, l_next = possible_neighbours[0]
            path.append( possible_neighbours[0] )
            visited.append( (k_curr,l_curr))
            maze.grid[k_next][l_next].visited = True

            visit_counter+= 1

            k_curr, l_curr = k_next, l_next

        else:
            if len( visited ) != 0:
                k_curr, l_curr = visited.pop()
                path.append( (k_curr,l_curr) )
            else:
                break
    for row in maze.grid:
        for cell in row:
            cell.visited = False

    print(f"Generating path for maze took {time.time() - begin_time}s.")
    maze.generation_path = path


class Maze(object):

    def __init__(self, num_rows, num_cols, id=0, algorithm = "dfs_backtrack"):
        self.num_cols = num_cols
        self.num_rows = num_rows
        self.id = id
        self.grid_size = num_rows*num_cols
        self.entry_coor = (0, 0)
        self.exit_coor = (num_cols-1, num_rows-1)
        self.generation_path = []
        self.solution_path = None
        self.initial_grid = self.generate_grid()
        self.grid = self.initial_grid
        self.generate_maze(algorithm, (0, 0))

    def generate_grid(self):
        grid = list()

        for i in range(self.num_rows):
            grid.append(list())

            for j in range(self.num_cols):
                grid[i].append(Cell(i, j))

        return grid

    def find_neighbours(self, cell_row, cell_col):
        neighbours = list()

        def check_neighbour(row, col):
            if row >= 0 and row < self.num_rows and col >= 0 and col < self.num_cols:
                neighbours.append((row, col))

        check_neighbour(cell_row-1, cell_col)
        check_neighbour(cell_row, cell_col+1)
        check_neighbour(cell_row+1, cell_col)
        check_neighbour(cell_row, cell_col-1)

        if len(neighbours) > 0:
            return neighbours

        else:
            return None  

    def _validate_neighbours_generate(self, neighbour_indices):

        neigh_list = [n for n in neighbour_indices if not self.grid[n[0]][n[1]].visited]

        if len(neigh_list) > 0:
            return neigh_list
        else:
            return None

    def validate_neighbours_solve(self, neighbour_indices, k, l, k_end, l_end, method = "fancy"):
        if method == "fancy":
            neigh_list = list()
            min_dist_to_target = 100000

            for k_n, l_n in neighbour_indices:
                if (not self.grid[k_n][l_n].visited
                        and not self.grid[k][l].is_walls_between(self.grid[k_n][l_n])):
                    dist_to_target = math.sqrt((k_n - k_end) ** 2 + (l_n - l_end) ** 2)

                    if (dist_to_target < min_dist_to_target):
                        min_dist_to_target = dist_to_target
                        min_neigh = (k_n, l_n)

            if "min_neigh" in locals():
                neigh_list.append(min_neigh)

        elif method == "brute-force":
            neigh_list = [n for n in neighbour_indices if not self.grid[n[0]][n[1]].visited
                          and not self.grid[k][l].is_walls_between(self.grid[n[0]][n[1]])]

        if len(neigh_list) > 0:
            return neigh_list
        else:
            return None

    def generate_maze(self, algorithm, start_coor=(0, 0)):

        if algorithm == "dfs_backtrack":
            depth_first_recursive_backtracker(self, start_coor)
        elif algorithm == "bin_tree":
            binary_tree(self, start_coor)



class Visualizer(object):
    def __init__(self, maze, cell_size, media_filename):
        self.maze = maze
        self.cell_size = cell_size
        self.height = maze.num_rows * cell_size
        self.width = maze.num_cols * cell_size
        self.ax = None
        self.lines = dict()
        self.squares = dict()
        self.media_filename = media_filename

    def set_media_filename(self, filename):
        self.media_filename = filename

    def show_maze(self):

        fig = self.configure_plot()

        self.plot_walls()

        # plt.show()

        if self.media_filename:
            fig.savefig("{}{}.png".format("parham-envs/parham_envs/resources/statics/", self.media_filename))

    def plot_walls(self):
        for i in range(self.maze.num_rows):
            for j in range(self.maze.num_cols):
                if self.maze.initial_grid[i][j].walls["top"]:
                    self.ax.plot([j*self.cell_size, (j+1)*self.cell_size],
                                 [i*self.cell_size, i*self.cell_size], color="k", linewidth=10)
                if self.maze.initial_grid[i][j].walls["right"]:
                    self.ax.plot([(j+1)*self.cell_size, (j+1)*self.cell_size],
                                 [i*self.cell_size, (i+1)*self.cell_size], color="k", linewidth=10)
                if self.maze.initial_grid[i][j].walls["bottom"]:
                    self.ax.plot([(j+1)*self.cell_size, j*self.cell_size],
                                 [(i+1)*self.cell_size, (i+1)*self.cell_size], color="k", linewidth=10)
                if self.maze.initial_grid[i][j].walls["left"]:
                    self.ax.plot([j*self.cell_size, j*self.cell_size],
                                 [(i+1)*self.cell_size, i*self.cell_size], color="k", linewidth=10)

    def configure_plot(self):
        fig = plt.figure(figsize = (7, 7*self.maze.num_rows/self.maze.num_cols))

        self.ax = plt.axes()

        self.ax.set_aspect("equal")

        self.ax.axes.get_xaxis().set_visible(False)
        self.ax.axes.get_yaxis().set_visible(False)

        self.ax.axis('off')

        
        return fig

    def show_maze_solution(self):
        fig = self.configure_plot()

        self.plot_walls()

        list_of_backtrackers = [path_element[0] for path_element in self.maze.solution_path if path_element[1]]

        circle_num = 0

        self.ax.add_patch(plt.Circle(((self.maze.solution_path[0][0][1] + 0.5)*self.cell_size,
                                      (self.maze.solution_path[0][0][0] + 0.5)*self.cell_size), 0.2*self.cell_size,
                                     fc=(0, circle_num/(len(self.maze.solution_path) - 2*len(list_of_backtrackers)),
                                         0), alpha=0.4))

        for i in range(1, self.maze.solution_path.__len__()):
            if self.maze.solution_path[i][0] not in list_of_backtrackers and\
                    self.maze.solution_path[i-1][0] not in list_of_backtrackers:
                circle_num += 1
                self.ax.add_patch(plt.Circle(((self.maze.solution_path[i][0][1] + 0.5)*self.cell_size,
                    (self.maze.solution_path[i][0][0] + 0.5)*self.cell_size), 0.2*self.cell_size,
                    fc = (0, circle_num/(len(self.maze.solution_path) - 2*len(list_of_backtrackers)), 0),
                     alpha = 0.4))

        # plt.show()

        if self.media_filename:
            fig.savefig("{}{}{}.png".format("parham_envs/resources/statics/", self.media_filename, "_solution"))


class MazeManager(object):

    def __init__(self):
        self.mazes = []
        self.media_name = ""
        self.quiet_mode = False

    def add_maze(self, row, col, id=0):
        if id is not 0:
            self.mazes.append(Maze(row, col, id))
        else:
            if len(self.mazes) < 1:
                self.mazes.append(Maze(row, col, 0))
            else:
                self.mazes.append(Maze(row, col, len(self.mazes) + 1))

        return self.mazes[-1]

    def add_existing_maze(self, maze, override=True):
        if self.check_matching_id(maze.id) is None:
            if override:
                if len(self.mazes) < 1:
                    maze.id = 0
                else:
                    maze.id = self.mazes.__len__()+1
        else:
            return False
        self.mazes.append(maze)
        return maze

    def get_maze(self, id):
        for maze in self.mazes:
            if maze.id == id:
                return maze
        print("Unable to locate maze")
        return None

    def get_mazes(self):
        return self.mazes

    def get_maze_count(self):
        return self.mazes.__len__()

    def solve_maze(self, maze_id, method, neighbor_method="fancy"):
        maze = self.get_maze(maze_id)
        if maze is None:
            print("Unable to locate maze. Exiting solver.")
            return None

        if method == "DepthFirstBacktracker":
            solver = DepthFirstBacktracker(maze, neighbor_method, self.quiet_mode)
            maze.solution_path = solver.solve()
        elif method == "BiDirectional":
            solver = BiDirectional(maze, neighbor_method, self.quiet_mode)
            maze.solution_path = solver.solve()
        elif method == "BreadthFirst":
            solver = BreadthFirst(maze, neighbor_method, self.quiet_mode)
            maze.solution_path = solver.solve()

    def show_maze(self, id, cell_size=1):
        vis = Visualizer(self.get_maze(id), cell_size, self.media_name)
        vis.show_maze()

    def show_generation_animation(self, id, cell_size=1):
        vis = Visualizer(self.get_maze(id), cell_size, self.media_name)
        vis.show_generation_animation()

    def show_solution(self, id, cell_size=1):
        vis = Visualizer(self.get_maze(id), cell_size, self.media_name)
        vis.show_maze_solution()

    def show_solution_animation(self, id, cell_size =1):
        vis = Visualizer(self.get_maze(id), cell_size, self.media_name)
        vis.animate_maze_solution()

    def check_matching_id(self, id):
        return next((maze for maze in self.mazes if maze .id == id), None)

    def set_filename(self, filename):

        self.media_name = filename

    def set_quiet_mode(self, enabled):
        self.quiet_mode=enabled


if __name__ == "__main__":

    manager = MazeManager()

    maze = manager.add_maze(10, 10)

    maze2 = Maze(10, 10)
    maze2 = manager.add_existing_maze(maze2)
    
    maze_binTree = Maze(10, 10, algorithm = "bin_tree")
    maze_binTree = manager.add_existing_maze(maze_binTree)
    manager.solve_maze(maze.id, "DepthFirstBacktracker")
    manager.set_filename("maze")

    manager.show_maze(maze.id)

    path = maze.solution_path
    print(path)
    img = cv2.imread('parham-envs/parham_envs/resources/statics/maze.png', cv2.IMREAD_GRAYSCALE)
    img[img>125] = 255
    img[img<=125] = 0
    img = 255-img
    img = img[95:-86, 95:-86]
    cv2.imwrite('parham-envs/parham_envs/resources/statics/maze.png', img)
    path = [(50*y, 50*x) for (x, y), _ in path]
    cv2.imshow('maze', img)
    cv2.waitKey()
    manager.show_solution(maze.id)



# import cv2

# def solve_maze(file_path, offset=20, check = 20, debug=False):
#     image = cv2.imread(file_path)
#     img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     x, y = img.shape
#     points = []
#     print(x, y)
#     current_x, current_y = offset, 5
#     direction = 'down'
#     done = False
#     min_distance = 6
    
#     for i in range(204):
#         if debug:
#           print('-'*100)
#           print(i)
#           cv2.circle(image, (current_y, current_x), 3, (0, 255, 0), 1)
#           cv2.imshow('test', image)
#           key = cv2.waitKey()
#           if key == ord('q') or key == 27:
#               break
#         points.append((-current_y, current_x))
        
#         if direction == 'down':
#             if debug:
#               print("turn left or not", img[current_x, current_y:current_y+check])
#             if sum(img[current_x, current_y:current_y+check]) > 0:
#                 if debug:
#                   print("direct or not", img[current_x:current_x+check, current_y])
#                 if sum(img[current_x:current_x+check, current_y]) == 0:
#                     current_x += offset
#                 else:
#                     direction = 'right'
#             else:
#                 direction = 'left'
#                 current_y += offset

#         elif direction == 'left':
#             if debug:
#               print("turn left or not", img[current_x-check:current_x, current_y])
#             if sum(img[current_x-check:current_x, current_y]) > 0:
#                 if debug:
#                   print("direct or not", img[current_x, current_y:current_y+check])
#                 if sum(img[current_x, current_y:current_y+check]) == 0:
#                     current_y += offset
#                 else:
#                     direction = 'down'
#             else:
#                 direction = 'up'
#                 current_x -= offset

#         elif direction == 'up':
#             if debug:
#               print("turn left or not", img[current_x, current_y-check:current_y])
#             if sum(img[current_x, current_y-check:current_y]) > 0:
#                 if debug:
#                   print("direct or not", img[current_x-check:current_x, current_y])
#                 if sum(img[current_x-check:current_x, current_y]) == 0:
#                     current_x -= offset
#                 else:
#                     direction = 'left'
#             else:
#                 direction = 'right'
#                 current_y -= offset

#         elif direction == 'right':
#             if debug:
#               print("turn left or not", img[current_x:current_x+check, current_y])
#             if sum(img[current_x:current_x+check, current_y]) > 0:
#                 if debug:
#                   print("direct or not", img[current_x, current_y-check:current_y])
#                 if sum(img[current_x, current_y-check:current_y]) == 0:
#                     current_y -= offset
#                 else:
#                     direction = 'up'
#             else:
#                 direction = 'down'
#                 current_x += offset
        
#         if debug:
#           print(direction)

#         if sum(img[current_x-min_distance:current_x, current_y]) > 0:
#             current_x += min_distance
#         if sum(img[current_x:current_x+min_distance, current_y]) > 0:
#             current_x -= min_distance
#         if sum(img[current_x, current_y-min_distance:current_y]) > 0:
#             current_y += min_distance
#         if sum(img[current_x, current_y:current_y+min_distance]) > 0:
#             current_y -= min_distance
#     return points