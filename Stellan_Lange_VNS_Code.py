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

    
    y_pred = model.predict(X_test_Fea_selc)

    
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



def neighbor_sol(solution, num_flip):
    best_array= []
    best_fitness = 0
    if num_flip == 1:
        #1neghbours
        #print("\n neighbour =1:")
        for i in range(len(solution)):
            if solution[i] == 0:
                solution[i] = 1
            elif solution[i] == 1:
                solution[i] = 0
            #print(solution)
            solution_fitness = calculateFitness(solution)
            if solution_fitness > best_fitness:
                best_array = list(solution)
                best_fitness = solution_fitness
            if solution[i] == 0:
                solution[i] = 1
            elif solution[i] == 1:
                solution[i] = 0
    elif num_flip == 2:
        #2neighbours
        #print("\n neighbour =2:")
        for i in range(len(solution)):
            if solution[i] == 0:
                solution[i] = 1
                #print(solution)
                for j in range(i+1, len(solution)):
                    if solution[j] == 0:
                        solution[j] = 1
                    elif solution[j] == 1:
                        solution[j] = 0
                    #print(solution)
                    solution_fitness = calculateFitness(solution)
                    if solution_fitness > best_fitness:
                        best_array = list(solution)
                        best_fitness = solution_fitness
                    if solution[j] == 0:
                        solution[j] = 1
                    elif solution[j] == 1:
                        solution[j] = 0
            elif solution[i] == 1:
                solution[i] = 0
                #print(solution)
                for j in range(i+1, len(solution)):
                    if solution[j] == 0:
                        solution[j] = 1
                    elif solution[j] == 1:
                        solution[j] = 0
                    #print(solution)
                    solution_fitness = calculateFitness(solution)
                    if solution_fitness > best_fitness:
                        best_array = list(solution)
                        best_fitness = solution_fitness
                    if solution[j] == 0:
                        solution[j] = 1
                    elif solution[j] == 1:
                        solution[j] = 0
            
            if solution[i] == 0:
                solution[i] = 1
            elif solution[i] == 1:
                solution[i] = 0

    elif num_flip == 3:
        #3neighbours
        #print("\n neighbour =3:")
        for i in range(len(solution)):
            if solution[i] == 0:
                solution[i] = 1
                #print(solution)
                for j in range(i+1, len(solution)):
                    if solution[j] == 0:
                        solution[j] = 1
                        for k in range(j+1, len(solution)):
                            if solution[k] == 0:
                                solution[k] = 1
                            elif solution[k] == 1:
                                solution[k] = 0
                            #print(solution)
                            solution_fitness = calculateFitness(solution)
                            if solution_fitness > best_fitness:
                                best_array = list(solution)
                                best_fitness = solution_fitness
                            if solution[k] == 0:
                                solution[k] = 1
                            elif solution[k] == 1:
                                solution[k] = 0
                    elif solution[j] == 1:
                        solution[j] = 0
                        for k in range(j+1, len(solution)):
                            if solution[k] == 0:
                                solution[k] = 1
                            elif solution[k] == 1:
                                solution[k] = 0
                            #print(solution)
                            solution_fitness = calculateFitness(solution)
                            if solution_fitness > best_fitness:
                                best_array = list(solution)
                                best_fitness = solution_fitness
                            if solution[k] == 0:
                                solution[k] = 1
                            elif solution[k] == 1:
                                solution[k] = 0
                    #print(solution)
                    if solution[j] == 0:
                        solution[j] = 1
                    elif solution[j] == 1:
                        solution[j] = 0
            elif solution[i] == 1:
                solution[i] = 0
                #print(solution)
                for j in range(i+1, len(solution)):
                    if solution[j] == 0:
                        solution[j] = 1
                        for k in range(j+1, len(solution)):
                            if solution[k] == 0:
                                solution[k] = 1
                            elif solution[k] == 1:
                                solution[k] = 0
                            #print(solution)
                            solution_fitness = calculateFitness(solution)
                            if solution_fitness > best_fitness:
                                best_array = list(solution)
                                best_fitness = solution_fitness
                            if solution[k] == 0:
                                solution[k] = 1
                            elif solution[k] == 1:
                                solution[k] = 0
                    elif solution[j] == 1:
                        solution[j] = 0
                        for k in range(j+1, len(solution)):
                            if solution[k] == 0:
                                solution[k] = 1
                            elif solution[k] == 1:
                                solution[k] = 0
                            #print(solution)
                            solution_fitness = calculateFitness(solution)
                            if solution_fitness > best_fitness:
                                best_array = list(solution)
                                best_fitness = solution_fitness
                            if solution[k] == 0:
                                solution[k] = 1
                            elif solution[k] == 1:
                                solution[k] = 0
                    #print(solution)
                    if solution[j] == 0:
                        solution[j] = 1
                    elif solution[j] == 1:
                        solution[j] = 0
            
            if solution[i] == 0:
                solution[i] = 1
            elif solution[i] == 1:
                solution[i] = 0
            
            
    #print("the best array;", best_array, "the best fitness:", best_fitness)
    return best_array, best_fitness

number_of_lapses = 10
fitness_values = []

for lap in range(number_of_lapses):
    print('-'*50)
    bestSolution = rand_sol(numFeatures = 13)
    bestSolutionFitness = calculateFitness(bestSolution) 
    FITNESS_CALCULATIONS_COUNTER = 1
    num_flips = 1  
    improved = False

    current_solution = bestSolution  
    current_solution_fitness = bestSolutionFitness

    while FITNESS_CALCULATIONS_COUNTER < MAX_FITNESS_CALCULATIONS: 
        best_array, best_fitness = neighbor_sol(current_solution, num_flips)
        neighbor_solution_fitness = calculateFitness(best_array)

        
        FITNESS_CALCULATIONS_COUNTER += 1 
        
        if best_fitness > 90:
            bestSolution = best_array
            bestSolutionFitness = best_fitness
            break

        if best_fitness > current_solution_fitness:
            current_solution = best_array 
            current_solution_fitness = best_fitness
            num_flips = 1
            improved = True
            print(f"Best solution fitness ( {FITNESS_CALCULATIONS_COUNTER} / 5000 ): {bestSolutionFitness}")
        
        else:
            if num_flips < 3:  
                num_flips += 1
                improved = False
            else:
                num_flips = 1
                current_solution = rand_sol(numFeatures = 13) 


    
            
        

    fitness_values.append(bestSolutionFitness)
    print(f"bestSolution: {bestSolution} Best fitness: {bestSolutionFitness}")

average_fitness = sum(fitness_values) / len(fitness_values)
print(f"Average Fitness over {number_of_lapses} runs: {average_fitness}")
