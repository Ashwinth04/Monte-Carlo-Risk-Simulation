#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <omp.h>

using namespace std;

#define N 500000

struct Asset {
    string name;
    double initialPrice;
    double expectedReturn;
    double volatility;
};

struct Portfolio {
    vector<Asset> assets;
    vector<double> weights;
    vector<vector<double>> correlationMatrix;
};

class MonteCarloSimulation {
private:
    Portfolio portfolio;
    int numSimulations;
    int timeSteps;
    double timeHorizon;
    
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
        
        simulatedReturns.resize(numSimulations);
    }
    
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
    
    vector<double> generateCorrelatedRandoms(mt19937& gen, normal_distribution<double>& dist,
                                                 const vector<vector<double>>& choleskyMatrix) {
        int numAssets = portfolio.assets.size();
        vector<double> uncorrelatedRandoms(numAssets);
        vector<double> correlatedRandoms(numAssets, 0.0);
        
        for (int i = 0; i < numAssets; i++) {
            uncorrelatedRandoms[i] = dist(gen);
        }
        
        for (int i = 0; i < numAssets; i++) {
            for (int j = 0; j <= i; j++) {
                correlatedRandoms[i] += choleskyMatrix[i][j] * uncorrelatedRandoms[j];
            }
        }
        
        return correlatedRandoms;
    }
    
    void runSimulation(int numThreads = 0) {
        if (numThreads > 0) {
            omp_set_num_threads(numThreads);
        }
        
        int numAssets = portfolio.assets.size();
        double dt = timeHorizon / timeSteps;
        
        vector<vector<double>> choleskyMatrix = choleskyDecomposition();
        
        vector<double> portfolioReturns(numSimulations);
        
        auto start = chrono::high_resolution_clock::now();
        
        #pragma omp parallel
        {
            unsigned int seed = omp_get_thread_num() * 100 + chrono::system_clock::now().time_since_epoch().count();
            mt19937 gen(seed);
            normal_distribution<double> dist(0.0, 1.0);
            
            #pragma omp for schedule(dynamic)
            for (int sim = 0; sim < numSimulations; sim++) {
                vector<double> currentPrices(numAssets);
                
                for (int asset = 0; asset < numAssets; asset++) {
                    currentPrices[asset] = portfolio.assets[asset].initialPrice;
                }
                
                for (int t = 0; t < timeSteps; t++) {
                    vector<double> correlatedRandoms = generateCorrelatedRandoms(gen, dist, choleskyMatrix);
                    
                    for (int asset = 0; asset < numAssets; asset++) {
                        double mu = portfolio.assets[asset].expectedReturn;
                        double sigma = portfolio.assets[asset].volatility;
                
                        double drift = (mu - 0.5 * sigma * sigma) * dt;
                        double diffusion = sigma * sqrt(dt) * correlatedRandoms[asset];
                        
                        currentPrices[asset] *= exp(drift + diffusion);
                    }
                }
                
                double initialPortfolioValue = 0.0;
                double finalPortfolioValue = 0.0;
                
                for (int asset = 0; asset < numAssets; asset++) {
                    initialPortfolioValue += portfolio.weights[asset] * portfolio.assets[asset].initialPrice;
                    finalPortfolioValue += portfolio.weights[asset] * currentPrices[asset];
                }
                
                portfolioReturns[sim] = (finalPortfolioValue - initialPortfolioValue) / initialPortfolioValue;
            }
        }
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        
        for (int sim = 0; sim < numSimulations; sim++) {
            simulatedReturns[sim].push_back(portfolioReturns[sim]);
        }
        
        cout << "Simulation completed in " << elapsed.count() << " seconds using " 
                  << omp_get_max_threads() << " threads." << endl;
    }
    
    double calculateVaR(double confidenceLevel) {
        vector<double> portfolioReturns;
        for (const auto& simReturn : simulatedReturns) {
            portfolioReturns.push_back(simReturn[0]);
        }
        
        sort(portfolioReturns.begin(), portfolioReturns.end());
        
        int index = static_cast<int>((1.0 - confidenceLevel) * numSimulations);
        
        return -portfolioReturns[index];
    }
    
    double calculateCVaR(double confidenceLevel) {
        vector<double> portfolioReturns;
        for (const auto& simReturn : simulatedReturns) {
            portfolioReturns.push_back(simReturn[0]);
        }
        
        sort(portfolioReturns.begin(), portfolioReturns.end());
        
        int index = static_cast<int>((1.0 - confidenceLevel) * numSimulations);
        
        double sum = 0.0;
        for (int i = 0; i < index; i++) {
            sum += portfolioReturns[i];
        }
        
        return -sum / index;
    }
    
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
    int numThreads = 0;
    if (argc > 1) {
        numThreads = atoi(argv[1]);
    }
    
    Asset asset1 = {"S&P 500", 4500.0, 0.08, 0.15};
    Asset asset2 = {"Tesla", 180.0, 0.12, 0.40};
    Asset asset3 = {"Bitcoin", 60000.0, 0.15, 0.65};
    
    vector<double> weights = {0.6, 0.2, 0.2};
    
    vector<vector<double>> correlationMatrix = {
        {1.0, 0.5, 0.3},
        {0.5, 1.0, 0.4},
        {0.3, 0.4, 1.0}
    };
    
    Portfolio portfolio = {
        {asset1, asset2, asset3},
        weights,
        correlationMatrix
    };
    
    int numSimulations = N;  // Increased for better demonstration of parallelism
    int timeSteps = 252;  // Daily steps for a year
    double timeHorizon = 1.0;  // 1 year
    
    MonteCarloSimulation simulation(portfolio, numSimulations, timeSteps, timeHorizon);
    
    cout << "Running Monte Carlo simulation with OpenMP..." << endl;
    simulation.runSimulation(numThreads);
    
    double confidenceLevel = 0.95;  // 95% confidence level
    simulation.printResults(confidenceLevel);
    
    return 0;
}