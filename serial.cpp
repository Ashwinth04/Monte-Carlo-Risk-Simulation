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
    
    mt19937 generator;
    normal_distribution<double> normalDist;
    
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
        
        unsigned seed = chrono::system_clock::now().time_since_epoch().count();
        generator.seed(seed);
        
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
    
    vector<double> generateCorrelatedRandoms() {
        int numAssets = portfolio.assets.size();
        vector<double> uncorrelatedRandoms(numAssets);
        vector<double> correlatedRandoms(numAssets, 0.0);
        
        for (int i = 0; i < numAssets; i++) {
            uncorrelatedRandoms[i] = normalDist(generator);
        }
        
        vector<vector<double>> choleskyMatrix = choleskyDecomposition();
        
        for (int i = 0; i < numAssets; i++) {
            for (int j = 0; j <= i; j++) {
                correlatedRandoms[i] += choleskyMatrix[i][j] * uncorrelatedRandoms[j];
            }
        }
        
        return correlatedRandoms;
    }
    
    void runSimulation() {
        int numAssets = portfolio.assets.size();
        
        double dt = timeHorizon / timeSteps;
        
        vector<vector<vector<double>>> allPrices(numSimulations, 
            vector<vector<double>>(timeSteps + 1, 
                vector<double>(numAssets)));
        
        for (int sim = 0; sim < numSimulations; sim++) {
            for (int asset = 0; asset < numAssets; asset++) {
                allPrices[sim][0][asset] = portfolio.assets[asset].initialPrice;
            }
        }
        
        for (int sim = 0; sim < numSimulations; sim++) {
            for (int t = 0; t < timeSteps; t++) {
                vector<double> correlatedRandoms = generateCorrelatedRandoms();
                
                for (int asset = 0; asset < numAssets; asset++) {
                    double mu = portfolio.assets[asset].expectedReturn;
                    double sigma = portfolio.assets[asset].volatility;
                    double currentPrice = allPrices[sim][t][asset];
                    
                    double drift = (mu - 0.5 * sigma * sigma) * dt;
                    double diffusion = sigma * sqrt(dt) * correlatedRandoms[asset];
                    
                    double nextPrice = currentPrice * exp(drift + diffusion);
                    allPrices[sim][t + 1][asset] = nextPrice;
                }
            }
            
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

int main() {
    auto start = high_resolution_clock::now();
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

    int numSimulations = N;
    int timeSteps = 252;  // Daily steps for a year
    double timeHorizon = 1.0;  // 1 year
    
    MonteCarloSimulation simulation(portfolio, numSimulations, timeSteps, timeHorizon);
    
    cout << "Running Monte Carlo simulation..." << endl;
    simulation.runSimulation();
    
    double confidenceLevel = 0.95;  // 95% confidence level
    simulation.printResults(confidenceLevel);
    auto end = high_resolution_clock::now();

    double time_taken = duration_cast<milliseconds>(end - start).count();
    cout << "Time taken: " << time_taken << " ms" << endl;

    return 0;
}