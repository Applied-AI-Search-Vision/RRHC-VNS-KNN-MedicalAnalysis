from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
import random



f = open("Assignment 3 medical_dataset.data")
dataset_X = []
dataset_y = []
line = " "
while line != "":
    line = f.readline()
    line = line[:-1]
    if line != "":
        line = line.split(",")
        floatList = []
        for i in range(len(line)):
            if i < len(line)-1:
                floatList.append(float(line[i]))
            else:
                value = float(line[i])
                if value == 0:
                    dataset_y.append(0)
                else:
                    dataset_y.append(1)
        dataset_X.append(floatList)
f.close()


X_train, X_test, y_train, y_test = train_test_split(dataset_X, dataset_y, test_size = 0.25, random_state = 0)


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


model = KNeighborsClassifier(n_neighbors = 3)


def calculateFitness(solution):
    fitness = 0

    
    X_train_Fea_selc = []
    X_test_Fea_selc = []
    for example in X_train:
        X_train_Fea_selc.append([a*b for a,b in zip(example,solution)])
    for example in X_test:
        X_test_Fea_selc.append([a*b for a,b in zip(example,solution)])

    model.fit(X_train_Fea_selc, y_train)

    # We predict the test cases
    y_pred = model.predict(X_test_Fea_selc)

    # We calculate the Accuracy
    cm = confusion_matrix(y_test, y_pred)
    TP = cm[0][0] 
    FP = cm[0][1] 
    TN = cm[1][1] 
    FN = cm[1][0]

    fitness = (TP + TN) / (TP + TN + FP + FN)

    return round(fitness *100,2)


MAX_FITNESS_CALCULATIONS = 5000

def rand_sol(numFeatures):
    solution = [random.randint(0,1) for _ in range(numFeatures)] 
    return solution

def neighbor_sols(current_solution): 
    neighbors = []
    for i in range(len(current_solution)):
        neighbor = current_solution.copy()
        if neighbor[i] == 0:
            neighbor[i] = 1
        else:
            neighbor[i] = 0
        neighbors.append(neighbor)
        
    return neighbors
#a test for the neighbor_sols function
current_solution = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
neighbors = neighbor_sols(current_solution)
for neighbor in neighbors:
    print(neighbor)
########################################

number_of_lapses = 10
fitness_values = []

for lap in range(number_of_lapses):
    print('-'*50)
    FITNESS_CALCULATIONS_COUNTER = 0
    bestSolution = rand_sol(numFeatures = 13) # our starting point for some global best and its purely randomized = not great
    bestSolutionFitness = calculateFitness(bestSolution)
    FITNESS_CALCULATIONS_COUNTER += 1

    current_solution = bestSolution 
    current_solution_fitness = bestSolutionFitness 

    while FITNESS_CALCULATIONS_COUNTER < MAX_FITNESS_CALCULATIONS:
        neighbor_solutions = neighbor_sols(current_solution)
        improved = False
        for every_neighbor in neighbor_solutions:
            neighbor_solution_fitness = calculateFitness(every_neighbor)
            FITNESS_CALCULATIONS_COUNTER += 1 

            if neighbor_solution_fitness > current_solution_fitness:
                current_solution = every_neighbor
                current_solution_fitness = neighbor_solution_fitness
                improved = True

                if neighbor_solution_fitness > bestSolutionFitness:
                    bestSolution = every_neighbor
                    bestSolutionFitness = neighbor_solution_fitness
                    print(f"Best solution fitness ( {FITNESS_CALCULATIONS_COUNTER} / 5000 ): {bestSolutionFitness}")

        if not improved:  #if we have not found a better neighbor solution, then we need to restart
            current_solution = rand_sol(numFeatures = 13)  # restart from a random position
            current_solution_fitness = calculateFitness(current_solution)
            FITNESS_CALCULATIONS_COUNTER += 1 

            if current_solution_fitness > bestSolutionFitness: 
                bestSolution = current_solution  #it will eventually generate a goal state as the initial state. {FROM BOOK}
                bestSolutionFitness = current_solution_fitness
                print(f"Best solution fitness( {FITNESS_CALCULATIONS_COUNTER} / 5000 ): {bestSolutionFitness}")

    fitness_values.append(bestSolutionFitness)
    print(f"bestSolution: {bestSolution} Best fitness: {bestSolutionFitness}")

average_fitness = sum(fitness_values) / len(fitness_values)
print(f"Average Fitness over {number_of_lapses} runs: {average_fitness}")
