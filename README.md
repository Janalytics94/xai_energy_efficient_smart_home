# Topic S2: Explainable Recommendation Systems in Energy‐Efficient Smart Home
## Seminar Applied Predictive Analytics at HU

<<<<<<< HEAD
Link to Abstract
https://docs.google.com/document/d/1jYaAYHeiy-8xz6jHZOEGybr_TQCPGwcelyqtFpljhhI/edit#heading=h.dbw1g9cog2rz

Link to other notes:
https://docs.google.com/document/d/1_xYo9eFa8QD25YOE8TClkJW2xLA_Od7NyhG98tfMO6w/edit#

=======
# Usage Docker Container

```
docker build . -t xai 
docker run -it --name xai -v $(pwd):/root/xai/ xai bash
```

    ## CODING ENVIRONMENT FOR OUR RECOMMENDER SYSTEM

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
    ├── data                                                            # REFIT household data, price data, REFIT readme
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


# INSTRUCTIONS FOR RECREATING OUR RESULTS

### Possible Additional data sets 
https://github.com/smda/smart-meter-data-catalog/blob/master/datasets.yaml

### Small Data Sets 
 -  Weather, Humidity etc Smart  
    Home Data Set: link: http://traces.cs.umass.edu/index.php/Smart/Smart
    description: "The goal of the Smart* project is to optimize home energy consumption. Available here is a wide variety of data collected from three real homes, including electrical (usage and generation), environmental (e.g., temperature and humidity), and operational (e.g., wall switch events). Also available is minute-level electricity usage data from 400+ anonymous homes. Please see the Smart* home page for general information about the project, or the Smart* Tools download page for software that was used in the collection of this data."

### BIG DATA SETS    
- Almanac of Minutely Power Dataset (AMPds):link: http://ampds.org
 description: "The AMPds dataset has been release to help load disaggregation/NILM and eco-feedback researchers test their algorithms, models,      systems, and prototypes. AMPds contains electricity, water, and natural gas measurements at one minute intervals — a total of 1,051,200 readings per meter for 2 years of monitoring. Weather data from Environment Canada\'s YVR weather station has also been added. This hourly weather data covers the same period of time as AMPds and includes a summary of climate normals observed from the years between 1981-2010. Utility billing data is also included for cost analyses."

-  Electricity Consumption & Occupancy data set (ECO):
   link: https://www.vs.inf.ethz.ch/res/show.html?what=eco-data
   description: "This website provides access to the ECO data set (Electricity Consumption and Occupancy). The ECO data set is a comprehensive data set for non-intrusive load monitoring and occupancy detection research. It was collected in 6 Swiss households over a period of 8 months. For each of the households, the ECO data set provides: 1 Hz aggregate consumption data. Each measurement contains data on current, voltage, and phase shift for each of the three phases in the household; 1 Hz plug-level data measured from selected appliances. Occupancy information measured through a tablet computer (manual labeling) and a passive infrared sensor (in some of the households). We make the ECO data set available to the research community. You may directly access the data set, but we always like to receive a short description on what you plan to do with the data via e-mail to Wilhelm Kleiminger."

- GREEND Dataset:
    link: https://sourceforge.net/projects/greend/
    description: "GREEND is an energy dataset containing power measurements collected from multiple households in Austria and Italy. It provides detailed energy profiles on a per device basis with a sampling rate of 1 Hz. We expect to regularly provide snapshots of the dataset as more data is recorded and measurement platforms deployed. The GREEND dataset is free to use in research and commercial applications. If you want to access the data, please fill out the brief form at http://goo.gl/rtXjxT which will eventually provide you with the credentials to open the dataset archive."





## Adding Data:
 - Due to file size restrictions, we did not include any of the REFIT: Electrical Load Measurements data (Murray et al. 2017)
 - These files can be accessed using the following link: https://www.doi.org/10.15129/9ab14b0e-19ac-4279-938f-27f643078cec
 - After downloading the clean household data needs to be copied to ./data


# REFERENCES

ENTSO-E (n.d.). Day-ahead prices (12.1.d) [online database]. Retrieved December 8, 2020, from https://transparency.entsoe.eu/transmission-domain/r2/dayAheadPrices/show

Murray, D., Stankovic, L., & Stankovic, V. (2017). An electrical load measurements datasetof united kingdom households from a two-year longitudinal study [data set]. Scientific Data.
>>>>>>> jana
