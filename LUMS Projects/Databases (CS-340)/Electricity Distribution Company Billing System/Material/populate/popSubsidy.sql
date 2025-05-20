
--Populate Subsidy Provider Table

INSERT INTO SubsidyProvider( ProviderID,  ProviderName)
	values ('GOP','Government of Punjab');



--Populate Subsidy Table

INSERT INTO Subsidy (SubsidyCode, ProviderID, ConnectionTypeCode, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, SubsidyDescription, RatePerUnit)
	values ('GOP-Protected-Cust1','GOP', 1, TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.27, 0.83, 'Punjab Govt. Subsidy for customers consuming between 200 and 500 units', 14);

INSERT INTO Subsidy (SubsidyCode, ProviderID, ConnectionTypeCode, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, SubsidyDescription, RatePerUnit)
	values ('GOP-Protected-Cust2','GOP', 2, TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.27, 0.83, 'Punjab Govt. Subsidy for customers consuming between 200 and 500 units', 14);


