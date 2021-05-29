# Topic S2: Explainable Recommendation Systems in Energy‐Efficient Smart Home
## Seminar Applied Predictive Analytics at HU

# Usage Docker Container

```
docker build . -t xai 
docker run -it --name xai -v $(pwd):/root/xai/ xai bash
```

CODING ENVIRONMENT FOR OUR RECOMMENDER SYSTEM

.
├── README.txt                                                  # this readme file
│
├── code.                                                       # agent notebooks + .py scripts
│   ├── Activity_Agent.ipynb
│   ├── Evaluation_Agent.ipynb
│   ├── Load_Agent.ipynb
│   ├── Preparation_Agent.ipynb
│   ├── Price_Agent.ipynb
│   ├── Recommendation_Agent.ipynb
│   ├── Usage_Agent.ipynb
│   ├── agents.py
│   └── helper_functions.py
│
├── data                                                        # REFIT household data, price data, REFIT readme
│   ├── CLEAN_House1.csv                                            # household data (Murray et al., 2017, household 1 to 10) 
│   ├── [...]                                                       # is not included, however required for evaluation
│   ├── CLEAN_House10.csv                                           
│   ├── REFIT_Readme.txt
│   └── Day-ahead Prices_201501010000-201601010000.csv              # day-ahead prices provided by ENTSO-E, n.d.
│
│
└── export                                                      # path for exporting configurations and intermediate results
    ├── 1_config.json                                               # configurations used for evaluating households 1 to 10
    ├── [...]
    └── 10_config.json


INSTRUCTIONS FOR RECREATING OUR RESULTS

Adding Data:
 - Due to file size restrictions, we did not include any of the REFIT: Electrical Load Measurements data (Murray et al. 2017)
 - These files can be accessed using the following link: https://www.doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec
 - After downloading the clean household data needs to be copied to ./data


REFERENCES

ENTSO-E (n.d.). Day-ahead prices (12.1.d) [online database]. Retrieved December 8, 2020, from https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show

Murray, D., Stankovic, L., & Stankovic, V. (2017). An electrical load measurements datasetof united kingdom households from a two-year longitudinal study [data set]. Scientific Data.