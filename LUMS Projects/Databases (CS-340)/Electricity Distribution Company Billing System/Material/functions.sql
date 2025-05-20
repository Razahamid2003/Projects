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
-- variable declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

END fun_compute_SubsidyAmount;

----------------------------------------------------------
-- 2.4.1 Function to generate Bill by inserting records in the Bill Table
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_Generate_Bill (
    p_BillID        IN NUMBER,
    p_ConnectionID  IN VARCHAR2,
    p_BillingMonth  IN NUMBER,
    p_BillingYear   IN NUMBER,
    p_BillIssueDate IN DATE
) RETURN NUMBER

IS
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

END fun_Generate_Bill;

----------------------------------------------------------
-- 2.4.2 Function for generating monthly bills of all consumers
----------------------------------------------------------
CREATE OR REPLACE FUNCTION fun_batch_Billing (
    p_BillingMonth  IN NUMBER,
    p_BillingYear   IN NUMBER,
    p_BillIssueDate IN DATE
) RETURN NUMBER

IS
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

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
-- varaible declarations

BEGIN
-- main processing logic

EXCEPTION
-- exception handling

END fun_adjust_Bill;