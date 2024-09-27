# Genetic_Algorithm_for_Travelling_Salesman_Problem<br><br>

<br>This repository contains my implementation of the Genetic Algorithm for the Travelling Salesman Problem. This implementation is part of a mandatory assignment for my CSCI-561 Foundations of Artificial Intelligence Course by Prof. Wei Min Shen at Viterbi School of Engineering, University of Southern California.<br><br>
There are multiple iterations of this implementation. Some of them are -<br>
1) Naive Genetic Algorithm Implementation
2) Efficient Genetic Algorithm Implementation
3) Optimized Genetic Algorithm using 2-opt
4) Highly Optimized Genetic Algorithm using Lin-Kernighan Heuristic<br><br>

## Problem Statement<br>
This is a programming assignment in which you will apply AI search/optimization techniques to a 3-dimensional Travelling-Salesman Problem (TSP). Conceptually speaking, the space of traveling is a set of “cities” located on some grid points with (x, y, z) locations in which your AI agent has to travel. Your agent can travel from any city to any city, and the distance between two cities is defined as the Euclidean distance between the two grid locations.<br><br>
<b>Input:</b> A file input.txt in the current directory of your program will be formatted as follows:<br><br>
 ● 1st line: A strictly positive 32-bit integer N, indicating the number of “city” locations in the 3D space.<br>
 ● Next N lines: Each line is a city coordinate represented by three non-negative 32-bit integers separated by one space character, for the grid location of the city.<br><br>
<b>Output:</b> Report your path of traveling to the cities, that is, the total distance traveled and N locations of the cities.<br><br>
Example:<br>
 ● 1st line: Computed distance of the path.<br>
 ● Next N+1 lines: Each line has three non-negative 32-bit integers separated by one space character indicating the city visited in order.<br><br>
<b>Note:</b> Your path ends at the start city. Hence, you will have N+1 lines.<br><br>

## 1) Naive Genetic Approach<br>
This is my first (partly blind) implementation of the Genetic Algorithm on the Travelling Salesman Problem. I call this implementation Naive because I was not thinking of optimization at all while writing the code. For example, there is no Euclidean Distance Matrix, The fitness value for each path is calculated by iteratively going through the path and getting the Euclidean distance between two cities. This computationally heavy task causes the overall algorithm to run slowly for each iteration. It is also not scalable. As the number of cities increases, this naive implementation takes more and more time. Even the solution (best path) given by this implementation may not be as optimal due to the randomness involved with the Genetic Algorithm. (I don't think I'll be uploading this code file as it is not formatted well.)<br><br>

## 2) Proper Genetic Algorithm Implementation<br>
This implementation of code is more efficient than the previous iteration. It is also more optimal. The use of <b>Distance Matrix</b> for faster lookup significantly helps to reduce computations for each iteration. Moreover, the use of <b>Mutation</b>, <b>Tournament Selection</b>, and <b>Elitism</b> improves the solution given by the algorithm. All these changes contribute to a much lower path cost and take us towards a better solution. However, due to the randomness effect or maybe some other reasons beyond my comprehension, I have still seen this implementation perform sub-optimally once in a while. This may be due to the the optimization getting stuck in a local minima.<br><br>

## 3) Optimized Genetic Algorithm using 2-opt<br>
This implementation adds the two opt technique to the basic Genetic Algorithm. While the basic idea remains the same, 2-opt significantly helps in reducing the path cost. By using 2-opt, we iterate through each edge of the given path, and we swap the nodes of that edge. If this improves the solution, we keep the change and move to the next edge and repeat this process until no improvement can be made. While providing significantly better solutions, 2-opt takes excessive computational time on larger problems. This can be solved by applying the 2-opt every n generations or limiting the number of swaps allowed for each individual. Even with these constraints, this technique performs very well, giving better solutions than the previous implementation.<br><br>

## 4) Highly Optimized Genetic Algorithm using the Lin-Kernighan Heuristic<br>
Applying the Lin-kernighan Heuristic significantly optimized the Genetic Algorithm upto such a point that it can even handle the large number of cities, in relatively minimal time. LKH is considered one of the best heuristics for TSP, often providing optimal or near-optimal solutions each time. It improves upon simpler heuristics like 2-opt by allowing more complex move types. It uses a variable k-opt move, where k is determined dynamically during the search. It can also make a sequence if move that temporarily lengthen the tour, allowing it to escape local minima. A sequence of moves is only accepted when it provides the best gain. Though, similar to 2-opt, I found it can be computationally expensive to execute k-opt swaps. Therefore, in my implementation I have limited k to {2,3,4,5}. And similar to previous implementation, I only applied LKH to every n generations. Genetic Algorithm using LKH has shown by far the best results for me so far. I will add my Comparison txt file as well to show some rudimentary results.<br>

## References<br>

1) Croes, G. A. (1958). A method for solving traveling-salesman problems. Operations Research, 6(6), 791-812.<br>
2) Lin, S., & Kernighan, B. W. (1973). An effective heuristic algorithm for the traveling-salesman problem. Operations Research, 21(2), 498-516.
3) Nguyen, H. D., Yoshihara, I., Yamamori, K., & Yasunaga, M. (2007). Implementation of an effective hybrid GA for large-scale traveling salesman problems. IEEE Transactions on Systems, Man, and Cybernetics, Part B (Cybernetics), 37(1), 92-99.
4) Baraglia, R., Hidalgo, J. I., & Perego, R. (2001). A hybrid heuristic for the traveling salesman problem. IEEE Transactions on Evolutionary Computation, 5(6), 613-622.
