
--Populate Tax Authority Table

INSERT INTO TaxAuthority ( TaxAuthorityID,  AuthorityName)
	values ('FBR','Federal Board of Revenue');

INSERT INTO TaxAuthority ( TaxAuthorityID,  AuthorityName)
	values ('GOP','Government of Punjab');

INSERT INTO TaxAuthority ( TaxAuthorityID,  AuthorityName)
	values ('DISCO','Distribution Company');

INSERT INTO TaxAuthority ( TaxAuthorityID,  AuthorityName)
	values ('PTV','Pakistan Television Corporation');



--Populate Tax Rates Table

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (1,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (2,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (3,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (4,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (11,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (12,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (13,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (14,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (15,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (16,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (21,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (22,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (23,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (24,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (25,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (26,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (27,'GST', 'FBR', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.17);


INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (1,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (2,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (3,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (4,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (11,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (12,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (13,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (14,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (15,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (16,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (21,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (22,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (23,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (24,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (25,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (26,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);

INSERT INTO TaxRates (ConnectionTypeCode, TaxType, TaxAuthorityID, StartDate, EndDate,  Rate)
	values (27,'Electricity Duty', 'GOP', TO_DATE('01-JAN-2023', 'DD-MON-YYYY'), TO_DATE('31-DEC-2024', 'DD-MON-YYYY'), 0.01);



