
--Populate Tariff Table

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0001', 1, 1, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.09, 'Residential Customer 5KW Slab 1 - Peak Hour', 21.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0002', 2, 1, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.33, 'Residential Customer 5KW Slab 1 -- Off Peak Hour', 14.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0003', 1, 1, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.09, 999999.9999, 'Residential Customer 5KW Slab 2 - Peak Hour', 25.0, 1323, 63);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0004', 2, 1, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.33, 999999.9999, 'Residential Customer 5KW Slab 2 -- Off Peak Hour', 20.0, 3318, 237);
  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0005', 1, 2, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.09, 'Residential Customer 10KW Slab 1 - Peak Hour', 31.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0006', 2, 2, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.33, 'Residential Customer 10KW Slab 1 -- Off Peak Hour', 21.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0007', 1, 2, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.09, 999999.9999, 'Residential Customer 10KW Slab 2 - Peak Hour', 41.0, 1953, 63);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0008', 2, 2, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.33, 999999.9999, 'Residential Customer 10KW Slab 2 -- Off Peak Hour', 30.0, 4977, 237);


INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0009', 1, 3, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.09, 'Residential Customer 15KW Slab 1 - Peak Hour', 31.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0010', 2, 3, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.33, 'Residential Customer 15KW Slab 1 -- Off Peak Hour', 21.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0011', 1, 3, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.09, 999999.9999, 'Residential Customer 15KW Slab 2 - Peak Hour', 41.0, 1953, 63);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0012', 2, 3, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.33, 999999.9999, 'Residential Customer 15KW Slab 2 -- Off Peak Hour', 30.0, 4977, 237);


INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0013', 1, 4, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.09, 'Residential Customer 20KW Slab 1 - Peak Hour', 31.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0014', 2, 4, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.33, 'Residential Customer 20KW Slab 1 -- Off Peak Hour', 21.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0015', 1, 4, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.09, 999999.9999, 'Residential Customer 20KW Slab 2 - Peak Hour', 41.0, 1953, 63);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Res-0016', 2, 4, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.33, 999999.9999, 'Residential Customer 20KW Slab 2 -- Off Peak Hour', 30.0, 4977, 237);
 


INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0001', 1, 11, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.27, 'Commercial Customer 5KW Slab 1 - Peak Hour', 63.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0002', 2, 11, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.99, 'Commercial Customer 5KW Slab 1 -- Off Peak Hour', 42.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0003', 1, 11, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.27, 999999.9999, 'Commercial Customer 5KW Slab 2 - Peak Hour', 75.0, 11907, 189);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0004', 2, 11, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.99, 999999.9999, 'Commercial Customer 5KW Slab 2 -- Off Peak Hour', 60.0, 29862, 711);



INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0005', 1, 12, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.27, 'Commercial Customer 10KW Slab 1 - Peak Hour', 63.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0006', 2, 12, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.99, 'Commercial Customer 10KW Slab 1 -- Off Peak Hour', 42.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0007', 1, 12, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.27, 999999.9999, 'Commercial Customer 10KW Slab 2 - Peak Hour', 75.0, 11907, 189);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0008', 2, 12, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.99, 999999.9999, 'Commercial Customer 10KW Slab 2 -- Off Peak Hour', 60.0, 29862, 711);



INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0009', 1, 13, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.27, 'Commercial Customer 15KW Slab 1 - Peak Hour', 63.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0010', 2, 13, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.99, 'Commercial Customer 15KW Slab 1 -- Off Peak Hour', 42.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0011', 1, 13, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.27, 999999.9999, 'Commercial Customer 15KW Slab 2 - Peak Hour', 75.0, 11907, 189);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0012', 2, 13, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.99, 999999.9999, 'Commercial Customer 15KW Slab 2 -- Off Peak Hour', 60.0, 29862, 711);


INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0013', 1, 14, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.27, 'Commercial Customer 20KW Slab 1 - Peak Hour', 63.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0014', 2, 14, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.99, 'Commercial Customer 20KW Slab 1 -- Off Peak Hour', 42.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0015', 1, 14, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.27, 999999.9999, 'Commercial Customer 20KW Slab 2 - Peak Hour', 75.0, 11907, 189);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0016', 2, 14, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.99, 999999.9999, 'Commercial Customer 20KW Slab 2 -- Off Peak Hour', 60.0, 29862, 711);


INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0017', 1, 15, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.27, 'Commercial Customer 25KW Slab 1 - Peak Hour', 63.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0018', 2, 15, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.99, 'Commercial Customer 25KW Slab 1 -- Off Peak Hour', 42.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0019', 1, 15, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.27, 999999.9999, 'Commercial Customer 25KW Slab 2 - Peak Hour', 75.0, 11907, 189);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0020', 2, 15, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.99, 999999.9999, 'Commercial Customer 25KW Slab 2 -- Off Peak Hour', 60.0, 29862, 711);


INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0021', 1, 16, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.27, 'Commercial Customer 30KW Slab 1 - Peak Hour', 63.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0022', 2, 16, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 0.99, 'Commercial Customer 30KW Slab 1 -- Off Peak Hour', 42.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0023', 1, 16, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.27, 999999.9999, 'Commercial Customer 30KW Slab 2 - Peak Hour', 75.0, 11907, 189);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Com-0024', 2, 16, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.99, 999999.9999, 'Commercial Customer 30KW Slab 2 -- Off Peak Hour', 60.0, 29862, 711);

   

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0001', 1, 21, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 1.08, 'Industry Customer 100KW Slab 1 - Peak Hour', 30.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0002', 2, 21, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 4.0, 'Industry Customer 100KW Slab 1 -- Off Peak Hour', 30.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0003', 1, 21, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 1.08, 999999.9999, 'Industry Customer 100KW Slab 2 - Peak Hour', 45.0, 22680, 756);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0004', 2, 21, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 4.0, 999999.9999, 'Industry Customer 100KW Slab 2 -- Off Peak Hour', 45.0, 85320, 2844);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0005', 1, 22, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 1.08, 'Industry Customer 150KW Slab 1 - Peak Hour', 30.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0006', 2, 22, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 4.0, 'Industry Customer 150KW Slab 1 -- Off Peak Hour', 30.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0007', 1, 22, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 1.08, 999999.9999, 'Industry Customer 150KW Slab 2 - Peak Hour', 45.0, 22680, 756);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0008', 2, 22, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 4.0, 999999.9999, 'Industry Customer 150KW Slab 2 -- Off Peak Hour', 45.0, 85320, 2844);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0009', 1, 23, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 1.08, 'Industry Customer 200KW Slab 1 - Peak Hour', 30.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0010', 2, 23, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 4.0, 'Industry Customer 200KW Slab 1 -- Off Peak Hour', 30.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0011', 1, 23, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 1.08, 999999.9999, 'Industry Customer 200KW Slab 2 - Peak Hour', 45.0, 22680, 756);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0012', 2, 23, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 4.0, 999999.9999, 'Industry Customer 200KW Slab 2 -- Off Peak Hour', 45.0, 85320, 2844);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0013', 1, 24, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 1.08, 'Industry Customer 250KW Slab 1 - Peak Hour', 30.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0014', 2, 24, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 4.0, 'Industry Customer 250KW Slab 1 -- Off Peak Hour', 30.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0015', 1, 24, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 1.08, 999999.9999, 'Industry Customer 250KW Slab 2 - Peak Hour', 45.0, 22680, 756);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0016', 2, 24, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 4.0, 999999.9999, 'Industry Customer 250KW Slab 2 -- Off Peak Hour', 45.0, 85320, 2844);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0017', 1, 25, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 1.08, 'Industry Customer 300KW Slab 1 - Peak Hour', 30.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0018', 2, 25, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 4.0, 'Industry Customer 300KW Slab 1 -- Off Peak Hour', 30.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0019', 1, 25, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 1.08, 999999.9999, 'Industry Customer 300KW Slab 2 - Peak Hour', 45.0, 22680, 756);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0020', 2, 25, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 4.0, 999999.9999, 'Industry Customer 300KW Slab 2 -- Off Peak Hour', 45.0, 85320, 2844);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0021', 1, 26, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 1.08, 'Industry Customer 350KW Slab 1 - Peak Hour', 30.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0022', 2, 26, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 4.0, 'Industry Customer 350KW Slab 1 -- Off Peak Hour', 30.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0023', 1, 26, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 1.08, 999999.9999, 'Industry Customer 350KW Slab 2 - Peak Hour', 45.0, 22680, 756);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0024', 2, 26, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 4.0, 999999.9999, 'Industry Customer 350KW Slab 2 -- Off Peak Hour', 45.0, 85320, 2844);


INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0025', 1, 27, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 1.08, 'Industry Customer 400KW Slab 1 - Peak Hour', 30.0, 0, 0);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0026', 2, 27, 1, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0, 4.0, 'Industry Customer 400KW Slab 1 -- Off Peak Hour', 30.0, 0, 0);

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0027', 1, 27, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 1.08, 999999.9999, 'Industry Customer 400KW Slab 2 - Peak Hour', 45.0, 22680, 756);  

INSERT INTO Tariff (TariffCode, TariffType, ConnectionTypeCode, Slab, StartDate, EndDate, ThresholdLow_perHour, ThresholdHigh_perHour, TarrifDescription, RatePerUnit, MinAmount, MinUnit)
	values ('T-Ind-0028', 2, 27, 2, TO_DATE('01-JUN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 4.0, 999999.9999, 'Industry Customer 400KW Slab 2 -- Off Peak Hour', 45.0, 85320, 2844);
