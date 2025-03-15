#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>

using namespace std;
using namespace chrono;


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
    
    // Random number generator
    mt19937 generator;
    normal_distribution<double> normalDist;
    
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
        timeHorizon(horizon),
        normalDist(0.0, 1.0) {
        
        // Seed the random number generator with current time
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        generator.seed(seed);
        
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
    vector<double> generateCorrelatedRandoms() {
        int numAssets = portfolio.assets.size();
        vector<double> uncorrelatedRandoms(numAssets);
        vector<double> correlatedRandoms(numAssets, 0.0);
        
        // Generate uncorrelated standard normal random numbers
        for (int i = 0; i < numAssets; i++) {
            uncorrelatedRandoms[i] = normalDist(generator);
        }
        
        // Transform uncorrelated randoms to correlated using Cholesky decomposition
        vector<vector<double>> choleskyMatrix = choleskyDecomposition();
        
        for (int i = 0; i < numAssets; i++) {
            for (int j = 0; j <= i; j++) {
                correlatedRandoms[i] += choleskyMatrix[i][j] * uncorrelatedRandoms[j];
            }
        }
        
        return correlatedRandoms;
    }
    
    // Run the Monte Carlo simulation
    void runSimulation() {
        int numAssets = portfolio.assets.size();
        
        // Calculate the time step size
        double dt = timeHorizon / timeSteps;
        
        // Initialize prices matrix to store all asset prices for all simulations
        vector<vector<vector<double>>> allPrices(numSimulations, 
            vector<vector<double>>(timeSteps + 1, 
                vector<double>(numAssets)));
        
        // Initialize all starting prices
        for (int sim = 0; sim < numSimulations; sim++) {
            for (int asset = 0; asset < numAssets; asset++) {
                allPrices[sim][0][asset] = portfolio.assets[asset].initialPrice;
            }
        }
        
        // Run simulations
        for (int sim = 0; sim < numSimulations; sim++) {
            for (int t = 0; t < timeSteps; t++) {
                vector<double> correlatedRandoms = generateCorrelatedRandoms();
                
                for (int asset = 0; asset < numAssets; asset++) {
                    double mu = portfolio.assets[asset].expectedReturn;
                    double sigma = portfolio.assets[asset].volatility;
                    double currentPrice = allPrices[sim][t][asset];
                    
                    // Geometric Brownian Motion formula
                    double drift = (mu - 0.5 * sigma * sigma) * dt;
                    double diffusion = sigma * sqrt(dt) * correlatedRandoms[asset];
                    
                    // Calculate next price
                    double nextPrice = currentPrice * exp(drift + diffusion);
                    allPrices[sim][t + 1][asset] = nextPrice;
                }
            }
            
            // Calculate portfolio return for this simulation
            double initialPortfolioValue = 0.0;
            double finalPortfolioValue = 0.0;
            
            for (int asset = 0; asset < numAssets; asset++) {
                initialPortfolioValue += portfolio.weights[asset] * allPrices[sim][0][asset];
                finalPortfolioValue += portfolio.weights[asset] * allPrices[sim][timeSteps][asset];
            }
            
            double portfolioReturn = (finalPortfolioValue - initialPortfolioValue) / initialPortfolioValue;
            simulatedReturns[sim].push_back(portfolioReturn);
        }
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

int main() {
    // Example usage
    
    // Create a portfolio with 3 assets
    auto start = high_resolution_clock::now();
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
    int numSimulations = N;
    int timeSteps = 252;  // Daily steps for a year
    double timeHorizon = 1.0;  // 1 year
    
    MonteCarloSimulation simulation(portfolio, numSimulations, timeSteps, timeHorizon);
    
    // Run the simulation
    cout << "Running Monte Carlo simulation..." << endl;
    simulation.runSimulation();
    
    // Print results
    double confidenceLevel = 0.95;  // 95% confidence level
    simulation.printResults(confidenceLevel);
    auto end = high_resolution_clock::now();

    double time_taken = duration_cast<milliseconds>(end - start).count();
    cout << "Time taken: " << time_taken << " ms" << endl;

    return 0;
}