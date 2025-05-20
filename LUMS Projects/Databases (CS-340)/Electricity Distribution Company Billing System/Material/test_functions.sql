-- Simple script to call your created functions

SET SERVEROUTPUT ON

SET LINESIZE 600

DECLARE
-- declare the local variables to use in the block here
    ret_val NUMBER;
    l_peakUnits NUMBER;
    l_offPeakUnits NUMBER;
    l_peakAmount NUMBER;
    l_offPeakAmount NUMBER;
BEGIN

    -- compute billing days
    ret_val := fun_compute_BillingDays('DISCO-RES-1115', 1, 2024);
    DBMS_OUTPUT.PUT_LINE('Billing Days: ' || ret_val);

    -- compute import peak units
    l_peakUnits := fun_compute_ImportPeakUnits('DISCO-RES-1115', 1, 2024);
    DBMS_OUTPUT.PUT_LINE('Import Peak Units: ' || l_peakUnits);

    -- compute import off-peak units
    l_offPeakUnits := fun_compute_ImportOffPeakUnits('DISCO-RES-1115', 1, 2024);
    DBMS_OUTPUT.PUT_LINE('Import Off-Peak Units: ' || l_offPeakUnits);

    -- compute export off-peak units
    ret_val := fun_compute_ExportOffPeakUnits('DISCO-RES-1115', 1, 2024);
    DBMS_OUTPUT.PUT_LINE('Export Off-Peak Units: ' || ret_val);

    -- compute peak amount
    l_peakAmount := fun_compute_PeakAmount('DISCO-RES-1115', 1, 2024, TO_DATE('07-FEB-2024', 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Amount for peak units: ' || l_peakAmount);

    -- compute off-peak amount
    l_offPeakAmount := fun_compute_OffPeakAmount('DISCO-RES-1115', 1, 2024, TO_DATE('07-FEB-2024', 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Off-Peak Amount: ' || l_offPeakAmount);

    -- compute tax amount
    ret_val := fun_compute_TaxAmount('DISCO-RES-1115', 1, 2024, TO_DATE('07-FEB-2024', 'DD-MON-YYYY'), l_peakAmount, l_offPeakAmount);
    DBMS_OUTPUT.PUT_LINE('Tax Amount: ' || ret_val);

    -- compute FixedFee
    ret_val := fun_compute_FixedFee('DISCO-RES-1115', 1, 2024, TO_DATE('07-FEB-2024', 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Fixed Fee: ' || ret_val);

    -- compute Arrears
    ret_val := fun_compute_Arrears('DISCO-RES-1115', 1, 2024, TO_DATE('07-FEB-2024', 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Arrears: ' || ret_val);

    -- compute SubsidyAmount
    ret_val := fun_compute_SubsidyAmount('DISCO-RES-1115', 1, 2024, TO_DATE('07-FEB-2024', 'DD-MON-YYYY'), l_peakUnits, l_offPeakUnits);
    DBMS_OUTPUT.PUT_LINE('Subsidy Amount: ' || ret_val);

    -- -- generate and insert bill
    -- ret_val := fun_generate_Bill('DISCO-RES-1115', 1, 2023);
    -- DBMS_OUTPUT.PUT_LINE('Bill Generated: ' || ret_val);

    -- generate monthly bills of all consumers
    ret_val := fun_batch_Billing(1, 2024, TO_DATE('07-FEB-2023', 'DD-MON-YYYY'));
    DBMS_OUTPUT.PUT_LINE('Batch Billing status: ' || ret_val);
    
    -- functions that also insert have not been called to avoid potential insertion of wrong data
END;

.

/
