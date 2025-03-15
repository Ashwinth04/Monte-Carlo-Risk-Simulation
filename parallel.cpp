#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>  // OpenMP header

using namespace std;

#define N 500000

// Define a structure to represent an asset
struct Asset {
    string name;
    double initialPrice;
    double expectedReturn;
    double volatility;
};

// Define a structure to represent a portfolio
struct Portfolio {
    vector<Asset> assets;
    vector<double> weights;
    vector<vector<double>> correlationMatrix;
};

// Class for Monte Carlo simulation
class MonteCarloSimulation {
private:
    Portfolio portfolio;
    int numSimulations;
    int timeSteps;
    double timeHorizon;
    
    // Matrix to store simulation results
    vector<vector<double>> simulatedReturns;
    
public:
    MonteCarloSimulation(
        const Portfolio& port, 
        int numSim, 
        int steps, 
        double horizon) : 
        portfolio(port), 
        numSimulations(numSim), 
        timeSteps(steps), 
        timeHorizon(horizon) {
        
        // Initialize the returns matrix
        simulatedReturns.resize(numSimulations);
    }
    
    // Perform Cholesky decomposition for correlated random numbers
    vector<vector<double>> choleskyDecomposition() {
        int n = portfolio.assets.size();
        vector<vector<double>> L(n, vector<double>(n, 0.0));
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j <= i; j++) {
                double sum = 0.0;
                
                if (j == i) {
                    for (int k = 0; k < j; k++) {
                        sum += L[j][k] * L[j][k];
                    }
                    L[j][j] = sqrt(portfolio.correlationMatrix[j][j] - sum);
                } else {
                    for (int k = 0; k < j; k++) {
                        sum += L[i][k] * L[j][k];
                    }
                    L[i][j] = (portfolio.correlationMatrix[i][j] - sum) / L[j][j];
                }
            }
        }
        
        return L;
    }
    
    // Generate correlated random numbers using Cholesky decomposition
    vector<double> generateCorrelatedRandoms(mt19937& gen, normal_distribution<double>& dist,
                                                 const vector<vector<double>>& choleskyMatrix) {
        int numAssets = portfolio.assets.size();
        vector<double> uncorrelatedRandoms(numAssets);
        vector<double> correlatedRandoms(numAssets, 0.0);
        
        // Generate uncorrelated standard normal random numbers
        for (int i = 0; i < numAssets; i++) {
            uncorrelatedRandoms[i] = dist(gen);
        }
        
        // Transform uncorrelated randoms to correlated using Cholesky decomposition
        for (int i = 0; i < numAssets; i++) {
            for (int j = 0; j <= i; j++) {
                correlatedRandoms[i] += choleskyMatrix[i][j] * uncorrelatedRandoms[j];
            }
        }
        
        return correlatedRandoms;
    }
    
    // Run the Monte Carlo simulation
    void runSimulation(int numThreads = 0) {
        // Set number of threads if specified, otherwise use default
        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        }
        
        int numAssets = portfolio.assets.size();
        double dt = timeHorizon / timeSteps;
        
        // Pre-compute the Cholesky matrix once
        vector<vector<double>> choleskyMatrix = choleskyDecomposition();
        
        // Initialize storage for all simulation results
        vector<double> portfolioReturns(numSimulations);
        
        // Start timing
        auto start = chrono::high_resolution_clock::now();
        
        // Run simulations in parallel
        #pragma omp parallel
        {
            // Create thread-local random number generator
            unsigned int seed = omp_get_thread_num() * 100 + chrono::system_clock::now().time_since_epoch().count();
            mt19937 gen(seed);
            normal_distribution<double> dist(0.0, 1.0);
            
            // Parallelize the simulation loop
            #pragma omp for schedule(dynamic)
            for (int sim = 0; sim < numSimulations; sim++) {
                // Array to hold current prices of all assets
                vector<double> currentPrices(numAssets);
                
                // Initialize starting prices
                for (int asset = 0; asset < numAssets; asset++) {
                    currentPrices[asset] = portfolio.assets[asset].initialPrice;
                }
                
                // Simulate price paths
                for (int t = 0; t < timeSteps; t++) {
                    vector<double> correlatedRandoms = generateCorrelatedRandoms(gen, dist, choleskyMatrix);
                    
                    for (int asset = 0; asset < numAssets; asset++) {
                        double mu = portfolio.assets[asset].expectedReturn;
                        double sigma = portfolio.assets[asset].volatility;
                        
                        // Geometric Brownian Motion formula
                        double drift = (mu - 0.5 * sigma * sigma) * dt;
                        double diffusion = sigma * sqrt(dt) * correlatedRandoms[asset];
                        
                        // Update price
                        currentPrices[asset] *= exp(drift + diffusion);
                    }
                }
                
                // Calculate portfolio return for this simulation
                double initialPortfolioValue = 0.0;
                double finalPortfolioValue = 0.0;
                
                for (int asset = 0; asset < numAssets; asset++) {
                    initialPortfolioValue += portfolio.weights[asset] * portfolio.assets[asset].initialPrice;
                    finalPortfolioValue += portfolio.weights[asset] * currentPrices[asset];
                }
                
                portfolioReturns[sim] = (finalPortfolioValue - initialPortfolioValue) / initialPortfolioValue;
            }
        }
        
        // End timing
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        
        // Store simulation results for risk metrics calculation
        for (int sim = 0; sim < numSimulations; sim++) {
            simulatedReturns[sim].push_back(portfolioReturns[sim]);
        }
        
        cout << "Simulation completed in " << elapsed.count() << " seconds using " 
                  << omp_get_max_threads() << " threads." << endl;
    }
    
    // Calculate Value at Risk (VaR)
    double calculateVaR(double confidenceLevel) {
        vector<double> portfolioReturns;
        for (const auto& simReturn : simulatedReturns) {
            portfolioReturns.push_back(simReturn[0]);
        }
        
        // Sort returns in ascending order
        sort(portfolioReturns.begin(), portfolioReturns.end());
        
        // Calculate the index for the VaR
        int index = static_cast<int>((1.0 - confidenceLevel) * numSimulations);
        
        // Return the negative of the return at that index (since VaR is a loss)
        return -portfolioReturns[index];
    }
    
    // Calculate Conditional Value at Risk (CVaR)
    double calculateCVaR(double confidenceLevel) {
        vector<double> portfolioReturns;
        for (const auto& simReturn : simulatedReturns) {
            portfolioReturns.push_back(simReturn[0]);
        }
        
        // Sort returns in ascending order
        sort(portfolioReturns.begin(), portfolioReturns.end());
        
        // Calculate the index for the VaR
        int index = static_cast<int>((1.0 - confidenceLevel) * numSimulations);
        
        // Calculate the average of all returns worse than VaR
        double sum = 0.0;
        for (int i = 0; i < index; i++) {
            sum += portfolioReturns[i];
        }
        
        // Return the negative of the average (since CVaR is a loss)
        return -sum / index;
    }
    
    // Print simulation results
    void printResults(double confidenceLevel) {
        double var = calculateVaR(confidenceLevel);
        double cvar = calculateCVaR(confidenceLevel);
        
        cout << "Risk Analysis Results:" << endl;
        cout << "Number of simulations: " << numSimulations << endl;
        cout << "Time horizon: " << timeHorizon << " years" << endl;
        cout << "Confidence level: " << confidenceLevel * 100 << "%" << endl;
        cout << "Value at Risk (VaR): " << fixed << setprecision(4) << var * 100 << "%" << endl;
        cout << "Conditional Value at Risk (CVaR): " << fixed << setprecision(4) << cvar * 100 << "%" << endl;
    }
};

int main(int argc, char* argv[]) {
    // Get number of threads from command line if provided
    int numThreads = 0;
    if (argc > 1) {
        numThreads = atoi(argv[1]);
    }
    
    // Create a portfolio with 3 assets
    Asset asset1 = {"S&P 500", 4500.0, 0.08, 0.15};
    Asset asset2 = {"Tesla", 180.0, 0.12, 0.40};
    Asset asset3 = {"Bitcoin", 60000.0, 0.15, 0.65};
    
    // Define weights (must sum to 1)
    vector<double> weights = {0.6, 0.2, 0.2};
    
    // Define correlation matrix
    vector<vector<double>> correlationMatrix = {
        {1.0, 0.5, 0.3},
        {0.5, 1.0, 0.4},
        {0.3, 0.4, 1.0}
    };
    
    // Create the portfolio
    Portfolio portfolio = {
        {asset1, asset2, asset3},
        weights,
        correlationMatrix
    };
    
    // Create a Monte Carlo simulation
    int numSimulations = N;  // Increased for better demonstration of parallelism
    int timeSteps = 252;  // Daily steps for a year
    double timeHorizon = 1.0;  // 1 year
    
    MonteCarloSimulation simulation(portfolio, numSimulations, timeSteps, timeHorizon);
    
    // Run the simulation with specified number of threads
    cout << "Running Monte Carlo simulation with OpenMP..." << endl;
    simulation.runSimulation(numThreads);
    
    // Print results
    double confidenceLevel = 0.95;  // 95% confidence level
    simulation.printResults(confidenceLevel);
    
    return 0;
}