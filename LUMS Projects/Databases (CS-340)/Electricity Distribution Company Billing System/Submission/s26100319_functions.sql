-- Name: <Your Name>
-- Roll No: <Your Roll No>
-- Section: <Your Section>

-- The file contains the template for the functions to be implemented in the assignment. DO NOT MODIFY THE FUNCTION SIGNATURES. Only need to add your implementation within the function bodies.

----------------------------------------------------------
-- 2.1.1 Function to compute billing days
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_compute_BillingDays (
    p_ConnectionID IN VARCHAR2,
    p_BillingMonth IN NUMBER,
    p_BillingYear  IN NUMBER
) RETURN NUMBER

IS
    v_last_reading_current_month TIMESTAMP;
    v_last_reading_previous_month TIMESTAMP;
    v_billing_days NUMBER;

BEGIN
    SELECT MAX(Timestamp) INTO v_last_reading_current_month
    FROM MeterReadings
    WHERE 
        ConnectionID = p_ConnectionID
        AND EXTRACT(MONTH FROM Timestamp) = p_BillingMonth
        AND EXTRACT(YEAR FROM Timestamp) = p_BillingYear;

    SELECT MAX(Timestamp) INTO v_last_reading_previous_month
    FROM MeterReadings
    WHERE 
        ConnectionID = p_ConnectionID
        AND EXTRACT(MONTH FROM Timestamp) = (p_BillingMonth - 1)
        AND EXTRACT(YEAR FROM Timestamp) = CASE 
                                               WHEN p_BillingMonth = 1 THEN p_BillingYear - 1
                                               ELSE p_BillingYear 
                                           END;

    v_billing_days := ROUND(v_last_reading_current_month - v_last_reading_previous_month);

    RETURN v_billing_days;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN NULL;
    WHEN OTHERS THEN
        RAISE_APPLICATION_ERROR(-20001, 'An error occurred while calculating billing days');

END fun_compute_BillingDays;

----------------------------------------------------------
-- 2.1.2 Function to compute Import_PeakUnits
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_compute_ImportPeakUnits (
    p_ConnectionID IN VARCHAR2,
    p_BillingMonth IN NUMBER,
    p_BillingYear  IN NUMBER
) RETURN NUMBER

IS
    v_curr_import_peak_reading NUMBER;
    v_prev_import_peak_reading NUMBER;
    v_import_peak_units NUMBER;

BEGIN
    SELECT MAX(ImportPeakReading) INTO v_curr_import_peak_reading
    FROM MeterReadings
    WHERE 
        ConnectionID = p_ConnectionID
        AND EXTRACT(MONTH FROM Timestamp) = p_BillingMonth
        AND EXTRACT(YEAR FROM Timestamp) = p_BillingYear;

    SELECT MAX(ImportPeakReading) INTO v_prev_import_peak_reading
    FROM MeterReadings
    WHERE 
        ConnectionID = p_ConnectionID
        AND EXTRACT(MONTH FROM Timestamp) = (p_BillingMonth - 1)
        AND EXTRACT(YEAR FROM Timestamp) = CASE 
                                               WHEN p_BillingMonth = 1 THEN p_BillingYear - 1
                                               ELSE p_BillingYear 
                                           END;

    v_import_peak_units := v_curr_import_peak_reading - v_prev_import_peak_reading;

    RETURN v_import_peak_units;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1;

END fun_compute_ImportPeakUnits;

----------------------------------------------------------
-- 2.1.3 Function to compute Import_OffPeakUnits
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_compute_ImportOffPeakUnits (
    p_ConnectionID IN VARCHAR2,
    p_BillingMonth IN NUMBER,
    p_BillingYear  IN NUMBER
) RETURN NUMBER

IS
    v_curr_import_offpeak_reading NUMBER;
    v_prev_import_offpeak_reading NUMBER;
    v_import_offpeak_units NUMBER;

BEGIN
    SELECT MAX(ImportOffPeakReading) INTO v_curr_import_offpeak_reading
    FROM MeterReadings
    WHERE 
        ConnectionID = p_ConnectionID
        AND EXTRACT(MONTH FROM Timestamp) = p_BillingMonth
        AND EXTRACT(YEAR FROM Timestamp) = p_BillingYear;

    SELECT MAX(ImportOffPeakReading) INTO v_prev_import_offpeak_reading
    FROM MeterReadings
    WHERE 
        ConnectionID = p_ConnectionID
        AND EXTRACT(MONTH FROM Timestamp) = (p_BillingMonth - 1)
        AND EXTRACT(YEAR FROM Timestamp) = CASE 
                                               WHEN p_BillingMonth = 1 THEN p_BillingYear - 1
                                               ELSE p_BillingYear 
                                           END;

    v_import_offpeak_units := v_curr_import_offpeak_reading - v_prev_import_offpeak_reading;

    RETURN v_import_offpeak_units;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1;

END fun_compute_ImportOffPeakUnits;

----------------------------------------------------------
-- 2.1.4 Function to compute Export_OffPeakUnits
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_compute_ExportOffPeakUnits (
    p_ConnectionID IN VARCHAR2,
    p_BillingMonth IN NUMBER,
    p_BillingYear  IN NUMBER
) RETURN NUMBER

IS
    v_curr_export_offpeak_reading NUMBER;
    v_prev_export_offpeak_reading NUMBER;
    v_export_offpeak_units NUMBER;

BEGIN
    SELECT MAX(ExportOffPeakReading) INTO v_curr_export_offpeak_reading
    FROM MeterReadings
    WHERE 
        ConnectionID = p_ConnectionID
        AND EXTRACT(MONTH FROM Timestamp) = p_BillingMonth
        AND EXTRACT(YEAR FROM Timestamp) = p_BillingYear;

    SELECT MAX(ExportOffPeakReading) INTO v_prev_export_offpeak_reading
    FROM MeterReadings
    WHERE 
        ConnectionID = p_ConnectionID
        AND EXTRACT(MONTH FROM Timestamp) = (p_BillingMonth - 1)
        AND EXTRACT(YEAR FROM Timestamp) = CASE 
                                               WHEN p_BillingMonth = 1 THEN p_BillingYear - 1
                                               ELSE p_BillingYear 
                                           END;

    v_export_offpeak_units := v_curr_export_offpeak_reading - v_prev_export_offpeak_reading;

    RETURN v_export_offpeak_units;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1;

END fun_compute_ExportOffPeakUnits;

----------------------------------------------------------
-- 2.2.1 Function to compute PeakAmount
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_compute_PeakAmount (
    p_ConnectionID  IN VARCHAR2,
    p_BillingMonth  IN NUMBER,
    p_BillingYear   IN NUMBER,
    p_BillIssueDate IN DATE
) RETURN NUMBER

IS
    v_billing_days NUMBER;
    v_peak_units_import NUMBER;
    v_min_units NUMBER;
    v_rate_per_unit NUMBER;
    v_min_amount NUMBER;
    v_additional_units_peak_import NUMBER;
    v_peak_amount_import NUMBER;
    v_peak_amount NUMBER;

BEGIN
    v_billing_days := fun_compute_BillingDays(p_ConnectionID, p_BillingMonth, p_BillingYear);
    
    v_peak_units_import := fun_compute_ImportPeakUnits(p_ConnectionID, p_BillingMonth, p_BillingYear);
    
    SELECT t.MinUnits, t.RatePerUnit, t.MinAmount 
    INTO v_min_units, v_rate_per_unit, v_min_amount
    FROM TariffRates t
    JOIN Connections c ON c.ConnectionTypeCode = t.ConnectionTypeCode
    WHERE c.ConnectionID = p_ConnectionID
      AND t.TariffType = 'Import Peak'
      AND t.EffectiveDate <= p_BillIssueDate
    ORDER BY t.EffectiveDate DESC
    FETCH FIRST 1 ROW ONLY;

    v_additional_units_peak_import := v_peak_units_import - ((v_min_units * v_billing_days) / 30);

    v_peak_amount_import := (v_additional_units_peak_import * v_rate_per_unit) 
                            + ((v_min_amount * v_billing_days) / 30);

    v_peak_amount := ROUND(v_peak_amount_import, 2);

    RETURN v_peak_amount;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1; 

END fun_compute_PeakAmount;

----------------------------------------------------------
-- 2.2.2 Function to compute OffPeakAmount
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_compute_OffPeakAmount (
    p_ConnectionID  IN VARCHAR2,
    p_BillingMonth  IN NUMBER,
    p_BillingYear   IN NUMBER,
    p_BillIssueDate IN DATE
) RETURN NUMBER

IS
    v_billing_days NUMBER;
    v_off_peak_units_import NUMBER;
    v_off_peak_units_export NUMBER;
    v_min_units_import NUMBER;
    v_rate_per_unit_import NUMBER;
    v_min_amount_import NUMBER;
    v_min_units_export NUMBER;
    v_rate_per_unit_export NUMBER;
    v_min_amount_export NUMBER;
    v_additional_units_off_peak_import NUMBER;
    v_additional_units_off_peak_export NUMBER;
    v_off_peak_amount_import NUMBER;
    v_off_peak_amount_export NUMBER;
    v_off_peak_amount NUMBER;

BEGIN
    v_billing_days := fun_compute_BillingDays(p_ConnectionID, p_BillingMonth, p_BillingYear);
    v_off_peak_units_import := fun_compute_ImportOffPeakUnits(p_ConnectionID, p_BillingMonth, p_BillingYear);
    v_off_peak_units_export := fun_compute_ExportOffPeakUnits(p_ConnectionID, p_BillingMonth, p_BillingYear);
    
    SELECT t.MinUnits, t.RatePerUnit, t.MinAmount
    INTO v_min_units_import, v_rate_per_unit_import, v_min_amount_import
    FROM TariffRates t
    JOIN Connections c ON c.ConnectionTypeCode = t.ConnectionTypeCode
    WHERE c.ConnectionID = p_ConnectionID
      AND t.TariffType = 'Import Off-Peak'
      AND t.EffectiveDate <= p_BillIssueDate
    ORDER BY t.EffectiveDate DESC
    FETCH FIRST 1 ROW ONLY;

    SELECT t.MinUnits, t.RatePerUnit, t.MinAmount
    INTO v_min_units_export, v_rate_per_unit_export, v_min_amount_export
    FROM TariffRates t
    JOIN Connections c ON c.ConnectionTypeCode = t.ConnectionTypeCode
    WHERE c.ConnectionID = p_ConnectionID
      AND t.TariffType = 'Export Off-Peak'
      AND t.EffectiveDate <= p_BillIssueDate
    ORDER BY t.EffectiveDate DESC
    FETCH FIRST 1 ROW ONLY;

    v_additional_units_off_peak_import := v_off_peak_units_import - ((v_min_units_import * v_billing_days) / 30);
    v_additional_units_off_peak_export := v_off_peak_units_export - ((v_min_units_export * v_billing_days) / 30);
    v_off_peak_amount_import := (v_additional_units_off_peak_import * v_rate_per_unit_import) + ((v_min_amount_import * v_billing_days) / 30);
    v_off_peak_amount_export := (v_additional_units_off_peak_export * v_rate_per_unit_export) + ((v_min_amount_export * v_billing_days) / 30);

    v_off_peak_amount := v_off_peak_amount_import - v_off_peak_amount_export;

    RETURN ROUND(v_off_peak_amount, 2);

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1;

END fun_compute_OffPeakAmount;

----------------------------------------------------------
-- 2.3.1 Function to compute TaxAmount
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_compute_TaxAmount (
    p_ConnectionID  IN VARCHAR2,
    p_BillingMonth  IN NUMBER,
    p_BillingYear   IN NUMBER,
    p_BillIssueDate IN DATE,
    p_PeakAmount    IN NUMBER,
    p_OffPeakAmount IN NUMBER
) RETURN NUMBER

IS
    v_tax_rate NUMBER;
    v_tax_amount NUMBER := 0;
    v_total_amount NUMBER;

BEGIN
    v_total_amount := p_PeakAmount + p_OffPeakAmount;

    FOR tax_rec IN (
        SELECT t.Rate
        FROM TaxRates t
        JOIN Connections c ON c.ConnectionTypeCode = t.ConnectionTypeCode
        WHERE c.ConnectionID = p_ConnectionID
          AND t.EffectiveDate <= p_BillIssueDate
        ORDER BY t.EffectiveDate DESC
    ) LOOP
        v_tax_amount := v_tax_amount + (v_total_amount * tax_rec.Rate);
    END LOOP;

    RETURN ROUND(v_tax_amount, 2);

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1;

END fun_compute_TaxAmount;

----------------------------------------------------------
-- 2.3.2 Function to compute FixedFee Amount
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_compute_FixedFee (
    p_ConnectionID  IN VARCHAR2,
    p_BillingMonth  IN NUMBER,
    p_BillingYear   IN NUMBER,
    p_BillIssueDate IN DATE
) RETURN NUMBER

IS
    v_fixed_fee NUMBER := 0;

BEGIN
    FOR fixed_charge_rec IN (
        SELECT fc.Amount
        FROM FixedCharges fc
        JOIN Connections c ON c.ConnectionTypeCode = fc.ConnectionTypeCode
        WHERE c.ConnectionID = p_ConnectionID
          AND fc.EffectiveDate <= p_BillIssueDate
        ORDER BY fc.EffectiveDate DESC
    ) LOOP
        v_fixed_fee := v_fixed_fee + fixed_charge_rec.Amount;
    END LOOP;

    RETURN ROUND(v_fixed_fee, 2);

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1;

END fun_computeFixedFee;

----------------------------------------------------------
-- 2.3.3 Function to compute Arrears
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_compute_Arrears (
    p_ConnectionID  IN VARCHAR2,
    p_BillingMonth  IN NUMBER,
    p_BillingYear   IN NUMBER,
    p_BillIssueDate IN DATE
) RETURN NUMBER

IS
    v_arrears NUMBER := 0;
    v_total_due NUMBER := 0;
    v_paid_amount NUMBER := 0;
    v_prev_billing_month NUMBER;
    v_prev_billing_year NUMBER;

BEGIN
    IF p_BillingMonth = 1 THEN
        v_prev_billing_month := 12;
        v_prev_billing_year := p_BillingYear - 1;
    ELSE
        v_prev_billing_month := p_BillingMonth - 1;
        v_prev_billing_year := p_BillingYear;
    END IF;

    SELECT b.TotalAmount_AfterDueDate, NVL(b.AmountPaid, 0)
    INTO v_total_due, v_paid_amount
    FROM Bill b
    WHERE b.ConnectionID = p_ConnectionID
      AND b.BillingMonth = v_prev_billing_month
      AND b.BillingYear = v_prev_billing_year;

    IF v_paid_amount = 0 THEN
        v_arrears := v_total_due;
    ELSE
        v_arrears := v_total_due - v_paid_amount;
    END IF;

    RETURN ROUND(v_arrears, 2);

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1;

END fun_compute_Arrears;

----------------------------------------------------------
-- 2.3.4 Function to compute SubsidyAmount
----------------------------------------------------------
function fun_compute_SubsidyAmount (
    p_ConnectionID       IN VARCHAR2,
    p_BillingMonth       IN NUMBER,
    p_BillingYear        IN NUMBER,
    p_BillIssueDate      IN DATE,
    p_ImportPeakUnits    IN NUMBER,
    p_ImportOffPeakUnits IN NUMBER
) RETURN NUMBER

IS
    v_BillingDays NUMBER;
    v_UnitPerHourSubsidy NUMBER;
    v_TotalSubsidyAmount NUMBER := 0;
    v_RatePerUnit NUMBER;
    v_SubsidyAmount NUMBER;
    
    CURSOR c_SubsidyRates IS
        SELECT RatePerUnit
        FROM Subsidy
        WHERE ConnectionType = (SELECT ConnectionType FROM Consumer WHERE ConnectionID = p_ConnectionID)
          AND EffectiveDate <= p_BillIssueDate
          AND ExpiryDate >= p_BillIssueDate;

BEGIN
    SELECT LAST_DAY(TO_DATE(p_BillingYear || '-' || p_BillingMonth || '-01', 'YYYY-MM-DD')) - 
           TO_DATE(p_BillingYear || '-' || p_BillingMonth || '-01', 'YYYY-MM-DD') + 1
    INTO v_BillingDays
    FROM DUAL;

    v_UnitPerHourSubsidy := (p_ImportPeakUnits + p_ImportOffPeakUnits) / (v_BillingDays * 24);

    FOR rate IN c_SubsidyRates LOOP
        v_SubsidyAmount := (v_UnitPerHourSubsidy * 24 * v_BillingDays) * rate.RatePerUnit;
        v_TotalSubsidyAmount := v_TotalSubsidyAmount + v_SubsidyAmount;
    END LOOP;

    RETURN ROUND(v_TotalSubsidyAmount, 2);

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1;

END fun_compute_SubsidyAmount;

----------------------------------------------------------
-- 2.4.1 Function to generate Bill by inserting records in the Bill Table
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_generate_Bill (
    p_BillID IN NUMBER,
    p_ConnectionID IN VARCHAR2,
    p_BillingMonth IN NUMBER,
    p_BillingYear IN NUMBER,
    p_BillIssueDate IN DATE
)
RETURN NUMBER
IS
    v_PeakAmount NUMBER;
    v_OffPeakAmount NUMBER;
    v_TaxAmount NUMBER;
    v_FixedFee NUMBER;
    v_Arrears NUMBER;
    v_SubsidyAmount NUMBER;
    v_AdjustmentAmount NUMBER := 0;
    v_TotalAmountBeforeDueDate NUMBER;
    v_TotalAmountAfterDueDate NUMBER;
    v_DueDate DATE;
BEGIN
    v_PeakAmount := fun_compute_PeakAmount(p_ConnectionID, p_BillingMonth, p_BillingYear, p_BillIssueDate);
    IF v_PeakAmount = -1 THEN RETURN -1; END IF;

    v_OffPeakAmount := fun_compute_OffPeakAmount(p_ConnectionID, p_BillingMonth, p_BillingYear, p_BillIssueDate);
    IF v_OffPeakAmount = -1 THEN RETURN -1; END IF;

    v_TaxAmount := fun_compute_TaxAmount(p_ConnectionID, p_BillingMonth, p_BillingYear, p_BillIssueDate, v_PeakAmount, v_OffPeakAmount);
    IF v_TaxAmount = -1 THEN RETURN -1; END IF;

    v_FixedFee := fun_compute_FixedFee(p_ConnectionID, p_BillingMonth, p_BillingYear, p_BillIssueDate);
    IF v_FixedFee = -1 THEN RETURN -1; END IF;

    v_Arrears := fun_compute_Arrears(p_ConnectionID, p_BillingMonth, p_BillingYear, p_BillIssueDate);
    IF v_Arrears = -1 THEN RETURN -1; END IF;

    v_SubsidyAmount := fun_compute_SubsidyAmount(p_ConnectionID, p_BillingMonth, p_BillingYear, p_BillIssueDate, v_PeakAmount, v_OffPeakAmount);
    IF v_SubsidyAmount = -1 THEN RETURN -1; END IF;

    v_DueDate := p_BillIssueDate + 10;

    v_TotalAmountBeforeDueDate := (v_PeakAmount + v_OffPeakAmount + v_TaxAmount + v_FixedFee + v_Arrears) - (v_SubsidyAmount + v_AdjustmentAmount);

    v_TotalAmountAfterDueDate := v_TotalAmountBeforeDueDate * 1.10;

    INSERT INTO Bill (BillID, ConnectionID, BillingMonth, BillingYear, BillIssueDate, DueDate,
                      TotalAmountBeforeDueDate, TotalAmountAfterDueDate, PeakAmount, OffPeakAmount,
                      TaxAmount, FixedFee, Arrears, SubsidyAmount, AdjustmentAmount)

    VALUES (p_BillID, p_ConnectionID, p_BillingMonth, p_BillingYear, p_BillIssueDate, v_DueDate,
            ROUND(v_TotalAmountBeforeDueDate, 2), ROUND(v_TotalAmountAfterDueDate, 2), ROUND(v_PeakAmount, 2),
            ROUND(v_OffPeakAmount, 2), ROUND(v_TaxAmount, 2), ROUND(v_FixedFee, 2), ROUND(v_Arrears, 2),
            ROUND(v_SubsidyAmount, 2), v_AdjustmentAmount);

    COMMIT;

    RETURN 1;

EXCEPTION
    WHEN OTHERS THEN
        ROLLBACK;
        RETURN -1;
END fun_generate_Bill;

----------------------------------------------------------
-- 2.4.2 Function for generating monthly bills of all consumers
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_batch_Billing (
    p_BillingMonth  IN NUMBER,
    p_BillingYear   IN NUMBER,
    p_BillIssueDate IN DATE
) RETURN NUMBER

IS
    v_BillID NUMBER;
    v_TotalGeneratedBills NUMBER := 0;
    v_Result NUMBER;
    v_Cursor SYS_REFCURSOR;
    v_ConnectionID VARCHAR2(50);

    CURSOR active_connections IS
        SELECT ConnectionID
        FROM Connections
        WHERE Status = 'Active';

BEGIN
    OPEN active_connections;
    LOOP
        FETCH active_connections INTO v_ConnectionID;
        EXIT WHEN active_connections%NOTFOUND;

        v_BillID := bill_id_seq.NEXTVAL;
        v_Result := fun_generate_Bill(
            p_BillID => v_BillID,
            p_ConnectionID => v_ConnectionID,
            p_BillingMonth => p_BillingMonth,
            p_BillingYear => p_BillingYear,
            p_BillIssueDate => p_BillIssueDate
        );

        IF v_Result = 1 THEN
            v_TotalGeneratedBills := v_TotalGeneratedBills + 1;
        END IF;
    END LOOP;

    CLOSE active_connections;

    IF v_TotalGeneratedBills > 0 THEN
        RETURN v_TotalGeneratedBills;
    ELSE
        RETURN -1;
    END IF;

EXCEPTION
    WHEN OTHERS THEN
        RETURN -1;

END fun_batch_Billing;

----------------------------------------------------------
-- 3.1.1 Function to process and record Payment
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_process_Payment (
    p_BillID          IN NUMBER,
    p_PaymentDate     IN DATE,
    p_PaymentMethodID IN NUMBER,
    p_AmountPaid      IN NUMBER
) RETURN NUMBER

IS
    v_TotalAmountAfterDueDate NUMBER;
    v_DueDate DATE;
    v_RemainingBalance NUMBER;
    v_PaymentStatus VARCHAR2(20);
    v_NewPaymentID NUMBER;
    e_BillNotFound EXCEPTION;

BEGIN
    SELECT TotalAmountAfterDueDate, DueDate
    INTO v_TotalAmountAfterDueDate, v_DueDate
    FROM Bill
    WHERE BillID = p_BILLID;

    v_RemainingBalance := v_TotalAmountAfterDueDate - p_AmountPaid;
    IF v_RemainingBalance <= 0 THEN
        v_PaymentStatus := 'FULLY PAID';
        v_RemainingBalance := 0;
    ELSE
        v_PaymentStatus := 'PARTIALLY PAID';
    END IF;

    INSERT INTO PaymentDetails (
        PaymentID, BillID, PaymentDate, PaymentStatus, PaymentMethodID, PaymentAmount
    ) VALUES (
        payment_details_seq.NEXTVAL,
        p_BILLID,
        p_PaymentDate,
        v_PaymentStatus,
        p_PaymentMethodID,
        p_AmountPaid
    ) RETURNING PaymentID INTO v_NewPaymentID;

    IF v_PaymentStatus = 'PARTIALLY PAID' THEN
        UPDATE Bill
        SET Arrears = v_RemainingBalance
        WHERE BillID = p_BILLID;
    ELSE
        UPDATE Bill
        SET Arrears = 0
        WHERE BillID = p_BILLID;
    END IF;

    RETURN v_NewPaymentID;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        RAISE_APPLICATION_ERROR(-20001, 'BillID not found.');
        RETURN -1;
    WHEN OTHERS THEN
        RETURN -1;

END fun_process_Payment;

----------------------------------------------------------
-- 4.1.1 Function to make Bill adjustment
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_adjust_Bill (
    p_AdjustmentID       IN NUMBER,
    p_BillID             IN NUMBER,
    p_AdjustmentDate     IN DATE,
    p_OfficerName        IN VARCHAR2,
    p_OfficerDesignation IN VARCHAR2,
    p_OriginalBillAmount IN NUMBER,
    p_AdjustmentAmount   IN NUMBER,
    p_AdjustmentReason   IN VARCHAR2
) RETURN NUMBER

IS
    v_PeakAmount NUMBER;
    v_OffPeakAmount NUMBER;
    v_TaxAmount NUMBER;
    v_FixedFee NUMBER;
    v_Arrears NUMBER;
    v_SubsidyAmount NUMBER;
    v_NewTotalAmountBeforeDueDate NUMBER;
    v_NewTotalAmountAfterDueDate NUMBER;
    v_BillIssueDate DATE;

BEGIN
    SELECT BillIssueDate INTO v_BillIssueDate
    FROM Bill
    WHERE BillID = p_BILLID;

    SELECT PeakAmount, OffPeakAmount, TaxAmount, FixedFee, Arrears, SubsidyAmount
    INTO v_PeakAmount, v_OffPeakAmount, v_TaxAmount, v_FixedFee, v_Arrears, v_SubsidyAmount
    FROM Bill
    WHERE BillID = p_BILLID;

    v_NewTotalAmountBeforeDueDate := (v_PeakAmount + v_OffPeakAmount + v_TaxAmount + v_FixedFee + v_Arrears) 
                                     - (v_SubsidyAmount + p_AdjustmentAmount);

    v_NewTotalAmountAfterDueDate := v_NewTotalAmountBeforeDueDate * 1.10;

    UPDATE Bill
    SET TotalAmountBeforeDueDate = ROUND(v_NewTotalAmountBeforeDueDate, 2),
        TotalAmountAfterDueDate = ROUND(v_NewTotalAmountAfterDueDate, 2),
        AdjustmentAmount = p_AdjustmentAmount
    WHERE BillID = p_BILLID;

    INSERT INTO BillAdjustment (
        AdjustmentID, BillID, AdjustmentDate, OfficerName, OfficerDesignation,
        OriginalBillAmount, AdjustmentAmount, AdjustmentReason
    )
    VALUES (
        p_AdjustmentID, p_BILLID, p_AdjustmentDate, p_OfficerName, 
        p_OfficerDesignation, p_OriginalBillAmount, p_AdjustmentAmount, 
        p_AdjustmentReason
    );

    COMMIT;
    RETURN 1;

EXCEPTION
    WHEN NO_DATA_FOUND THEN
        ROLLBACK;
        RETURN -1;
    WHEN OTHERS THEN
        ROLLBACK;
        RETURN -1;

END fun_adjust_Bill;