# Python---What-influence-the-GHG-Emissions-in-Brampton-City-
What influence the GHG Emissions in Brampton City? 


Research/Project Questions: 
What are the key factors associates with the GHG emissions in the public buildings of Brampton City?
 Literature Review 
Globally, almost 80% of GHG emissions from human sources come from the burning of fossil fuels and industrial processes. In 2015, about 26% of Canada's total GHG emissions came from the oil and gas sector, 24% from transportation, 11% from electricity generation and 12% from buildings. Canada aims to reduce its GHG emissions to 30% below its 2005 emission levels by 2030.
Data Description  
The dataset retrieved from Brampton City Catalog and presents data on Energy Consumption and Greenhouse Gas Emissions. The datasets include annual electricity and natural gas consumption and associated greenhouse gas emissions at City operated facilities. The dependent variable is the estimates of GHG emissions between 2014 and 2017. The independent variables selected for this project were: 'INTERNAL_GROSS_AREA_SQ_FT',  'cWEEKLY_HOURS_OF_OPERATION',  'ELECTRICITY_KWH',  'NATURAL_GAS_M3',  'TOTAL_ENERGY_EKWH', 'ENERGY_INTENSITY_EKWH_SQ_FT',    'GHG_EMISSIONS_KG', 'GHG_INTENSITY_KG_SQ_FT', 'REPORT_YEAR', 'SITE_NAME', 'SITE_TYPE', 'ADDRESS'.
Co linearity: It was suspected by first inspection of the initial OLS model. VIF test ran to choose the Independent variables.  
VIF for ENERGY_INTENSITY_EKWH_SQ_FT + GHG_INTENSITY_KG_SQ_FT reflect obvious co linearity
 
Lowest VIF avoiding scale 5-10 is: GHG_EMISSIONS_KG  ~  ELECTRICITY_KWH + NATURAL_GAS_M3 + TOTAL_ENERGY_EKWH +  REPORT_YEAR
 
Results: Green House Gas emissions are only significantly influenced with the electricity consumption, the amount of natural consumed gas, the total energy consumption, and the year of the consumption in Brampton City buildings. All models reflect high score although co-linearity inspected. Further investigation of the scores may require. 
Model	Model Score	GridSearch Score
Linear Regression 	0.919	
Polynomial Regression 	0.986	0.998
Random Forest 	0.988	0.988



 

 

 

 
References
1.	 Geohub.Brampton data sets:  http://geohub.brampton.ca/datasets//energy-consumption-and-greenhouse-gas-emissions 
2.	 Ontario Regulation 397/11: https://www.ontario.ca/laws/regulation/r11397 
3.	 Greenhouse Gas Reporting Program (GHGRP) : https://open.canada.ca/en/external/comment/3476#comment-3476 
4.	 City of Brampton Grow Green Action Plan: http://www.brampton.ca/EN/residents/GrowGreen/Pages/Energy.aspx 
5.	http://www.brampton.ca/EN/residents/GrowGreen/Documents/Action_Plan_Energy.pdf 
6.	Implications of Changing Climate:  https://www.nrcan.gc.ca/environment/resources/publications/impacts-adaptation/reports/assessments/2008/ch3/10325 
7.	Economic analysis of the Pan-Canadian Framework

