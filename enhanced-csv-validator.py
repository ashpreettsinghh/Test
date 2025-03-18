import pandas as pd
import logging
import os
from typing import List, Dict, Tuple, Optional, Union, Any
import re
import glob


class CSVValidator:
    """
    A class to validate CSV files with numeric data and total rows/columns.
    
    This validator checks:
    1. If a total row exists and matches the sum of other rows
    2. If the row-wise balance matches the specified total column
    3. If the total column's total matches the sum of other total column values
    """
    
    # Patterns to identify total rows in data
    TOTAL_ROW_PATTERNS = [
        r'total',
        r'totals',
        r'Total (LCY)',
        r'Grand Totals',
        r'ReportTotals',
    ]
    
    # Column mapping patterns based on the image
    COLUMN_MAPPINGS = {
        # Balance column patterns
        "BALANCE_PATTERNS": [
            r"Balance Due", r"O/S Amt", r"Total Debts", r"Total\(Base\)", 
            r"Balance\(Source\)", r"Outstanding Balance", r"Current Ba", r"Total O/S"
        ],
        
        # Current period patterns
        "CURRENT_PATTERNS": [
            r"< 30 days", r"0 - 30", r"0-30", r"< 1 Month", r"MONTH", 
            r"dd/mm/yyyy..dd/mm/yyyy", r"-0", r"Up to 30 days", r"Current Bal"
        ],
        
        # Period 1 patterns
        "PERIOD1_PATTERNS": [
            r"Period 1", r"Month-1", r"< 60 days", r"1 Month", r"31 - 60", r"31-60", 
            r"1 Month Bal", r"Mth 1", r"dd/mm/yyyy \(31\)", r"1-30", r"30 Days", 
            r"31 to 60 days", r"1 Month Bal"
        ],
        
        # Period 2 patterns
        "PERIOD2_PATTERNS": [
            r"Period 2", r"Month-2", r"< 90 days", r"2 Mths", r"2 Months", 
            r"Over 60 Days", r"MONTH", r"MON'YEAR", r"dd/mm/yyyy..dd/mm/yyyy", 
            r"60 Days", r"61 to 90 days", r"2 Month Bal", r"61-90", r"Mth 2", 
            r"dd/mm/yyyy \(62\)"
        ],
        
        # Period 3 patterns
        "PERIOD3_PATTERNS": [
            r"Period 3", r"Month-3", r"3 Mths", r"3 Months", r"Over 90 Days", 
            r"MONTH", r"MON'YEAR", r"dd/mm/yyyy..dd/mm/yyyy", r"90 Days", 
            r"91\+ days", r"3 Month Bal", r"Mth 3", r"dd/mm/yyyy \(93\)"
        ],
        
        # Period 4 patterns
        "PERIOD4_PATTERNS": [
            r"Month-4", r"121 - 150", r"4 Months", r"91-119", r"4 Month Bal", 
            r"91 and over", r"4 Mths", r"3 Months\+"
        ],
        
        # Older periods patterns
        "OLDER_PATTERNS": [
            r"181\+", r"120 days\+", r"Month 6\+", r"\+120 days", 
            r"61 and Over", r"150 Days\+", r"Oldest", r"Before MONTH", 
            r"MONTH & Older", r"4 Months\+", r"Before dd/mm/yyyy", r"After 122 days", 
            r"121-", r"120\+", r"61 and Over", r"4 months \+ 5 Mths Bal", 
            r"\> 120", r"Mth4\+", r"Over 4 Months", r"over 90", 
            r"Before dd/mm/yyyy\(>93\)", r"181\+"
        ],
        
        # Credit limit patterns
        "CREDIT_LIMIT_PATTERNS": [
            r"C. Limit", r"Cr.Limit"
        ]
    }
    
    def __init__(
        self, 
        log_file: str = "data_validation.log", 
        log_level: int = logging.INFO,
        tolerance: float = 0.01
    ):
        """
        Initialize the CSV validator.
        
        Args:
            log_file: Path to log file
            log_level: Logging level
            tolerance: Tolerance for floating point comparison
        """
        self.setup_logging(log_file, log_level)
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
    
    def setup_logging(self, log_file: str, log_level: int) -> None:
        """Set up logging configuration."""
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        logging.basicConfig(
            filename=log_file,
            level=log_level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        
        # Add console handler for immediate feedback
        console = logging.StreamHandler()
        console.setLevel(log_level)
        formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')
        console.setFormatter(formatter)
        logging.getLogger('').addHandler(console)
    
    @staticmethod
    def clean_numeric_string(value: Any) -> str:
        """
        Clean and standardize numeric strings for conversion.
        
        Args:
            value: The value to clean
            
        Returns:
            Cleaned string ready for numeric conversion
        """
        if pd.isna(value) or value is None or value == "" or value == '-':
            return "0.00"
        
        s = str(value).strip()
        
        # Handle parentheses format (123) -> -123
        if s.startswith('-(') and s.endswith(')'):
            s = '-' + s[2:-1]
        elif s.startswith('(') and s.endswith(')'):
            s = '-' + s[1:-1]
            
        # Remove commas, dollar signs, and other non-numeric characters
        s = re.sub(r'[,$\s]', '', s)
        
        return s
    
    def get_numeric_dataframe(self, df: pd.DataFrame, columns_to_convert: List[str]) -> pd.DataFrame:
        """
        Convert specified columns to numeric values.
        
        Args:
            df: Original dataframe
            columns_to_convert: List of columns to convert to numeric
            
        Returns:
            DataFrame with numeric conversions applied
        """
        numeric_df = df.copy()
        
        for col in columns_to_convert:
            if col in numeric_df.columns:
                try:
                    # Clean the strings and convert to float
                    numeric_df[col] = numeric_df[col].apply(self.clean_numeric_string)
                    numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
                    
                    # Ensure we're dealing with float values, not integers
                    if not pd.api.types.is_float_dtype(numeric_df[col]):
                        numeric_df[col] = numeric_df[col].astype(float)
                except Exception as e:
                    self.logger.error(f"Error converting column '{col}' to numeric: {str(e)}")
        
        return numeric_df
    
    def find_total_row(self, df: pd.DataFrame, numeric_columns: List[str] = None) -> Tuple[bool, Optional[int]]:
        """
        Find if the dataframe has a total row and return its index.
        This method checks for two conditions:
        1. If any cell in the row contains total patterns
        2. If all string columns are None or empty while numeric columns have values
        
        Args:
            df: DataFrame to search
            numeric_columns: List of columns known to contain numeric data
            
        Returns:
            Tuple of (is_total_found, total_row_index)
        """
        # First try the pattern matching approach
        for idx, row in df.iloc[::-1].iterrows():
            # Convert row to string and check if any cell contains total patterns
            row_str = row.astype(str).str.lower()
            
            for pattern in self.TOTAL_ROW_PATTERNS:
                if row_str.str.contains(pattern).any():
                    self.logger.info(f"Total row found at index {idx} based on pattern matching")
                    return True, idx
        
        # If pattern matching fails, try the empty string columns approach
        if numeric_columns:
            # Get non-numeric columns (potentially string columns)
            all_columns = df.columns.tolist()
            string_columns = [col for col in all_columns if col not in numeric_columns]
            
            for idx, row in df.iloc[::-1].iterrows():
                # Check if all string columns are empty or None
                string_columns_empty = all(
                    pd.isna(row[col]) or str(row[col]).strip() == "" 
                    for col in string_columns
                )
                
                # Check if numeric columns have values (not all are empty)
                numeric_columns_have_values = not all(
                    pd.isna(row[col]) or str(row[col]).strip() == "" 
                    for col in numeric_columns
                )
                
                # If string columns are empty and numeric columns have values, it's likely a total row
                if string_columns_empty and numeric_columns_have_values:
                    self.logger.info(f"Total row found at index {idx} based on column content analysis")
                    return True, idx
        
        self.logger.warning("No total row found in the dataset")
        return False, None
    
    def validate_column_totals(
        self, 
        numeric_df: pd.DataFrame,
        total_columns: List[str],
        total_row_index: int
    ) -> Dict[str, bool]:
        """
        Validate if column totals match the provided total row.
        
        Args:
            numeric_df: DataFrame with numeric data
            total_columns: Columns to validate totals for
            total_row_index: Index of the total row
            
        Returns:
            Dictionary with validation results per column
        """
        validation_results = {}
        
        # Only include data rows, excluding the total row itself
        data_rows = numeric_df.loc[:total_row_index-1]
        
        # Calculate sums for each column
        computed_totals = data_rows[total_columns].sum()
        provided_totals = numeric_df.loc[total_row_index, total_columns]
        
        for col in total_columns:
            computed = computed_totals[col]
            provided = provided_totals[col]
            
            # Both are NaN - consider it valid
            if pd.isna(computed) and pd.isna(provided):
                is_valid = True
            # One is NaN but other isn't - invalid
            elif pd.isna(computed) or pd.isna(provided):
                is_valid = False
                self.logger.error(
                    f"Column total mismatch in '{col}': "
                    f"Expected {provided}, Computed {computed}"
                )
            # Both are numbers - check with tolerance
            else:
                is_valid = abs(computed - provided) <= self.tolerance
                if not is_valid:
                    self.logger.error(
                        f"Column total mismatch in '{col}': "
                        f"Expected {provided}, Computed {computed}, "
                        f"Difference: {abs(computed - provided)}"
                    )
            
            validation_results[col] = is_valid
        
        return validation_results
    
    def validate_row_balances(
        self,
        numeric_df: pd.DataFrame,
        balance_columns: List[str],
        total_column: str,
        total_row_index: Optional[int] = None
    ) -> List[Optional[bool]]:
        """
        Validate if row-wise sums match the total column.
        
        Args:
            numeric_df: DataFrame with numeric data
            balance_columns: Columns to sum for each row
            total_column: Column containing the expected total
            total_row_index: Index of the total row to skip (if any)
            
        Returns:
            List of validation results per row
        """
        row_validation_results = []
        
        for i in range(len(numeric_df)):
            # Skip the total row if specified
            if total_row_index is not None and i >= total_row_index:
                row_validation_results.append(None)
                continue
            
            # Sum the balance columns for the row
            row_sum = numeric_df.loc[i, balance_columns].sum()
            actual_balance = numeric_df.loc[i, total_column]
            
            # Handle both NaN cases
            if pd.isna(row_sum) and pd.isna(actual_balance):
                is_valid = True
            # Handle one NaN case
            elif pd.isna(row_sum) or pd.isna(actual_balance):
                is_valid = False
                self.logger.warning(
                    f"Row {i+1} has a balance mismatch: "
                    f"Expected {actual_balance}, Computed {row_sum}"
                )
            # Handle numeric comparison with tolerance
            else:
                is_valid = abs(row_sum - actual_balance) <= self.tolerance
                if not is_valid:
                    self.logger.warning(
                        f"Row {i+1} has a balance mismatch: "
                        f"Expected {actual_balance}, Computed {row_sum}, "
                        f"Difference: {abs(row_sum - actual_balance)}"
                    )
            
            row_validation_results.append(is_valid)
        
        return row_validation_results
    
    def validate_total_column_total(
        self,
        numeric_df: pd.DataFrame,
        total_column: str,
        total_row_index: int
    ) -> bool:
        """
        Validate if the total column's total value matches the sum of other values in the total column.
        
        Args:
            numeric_df: DataFrame with numeric data
            total_column: The total column to validate
            total_row_index: Index of the total row
            
        Returns:
            Boolean indicating if total column's total is valid
        """
        # Sum all values in the total column except the total row
        total_col_sum = numeric_df.loc[:total_row_index-1, total_column].sum()
        total_col_total = numeric_df.loc[total_row_index, total_column]
        
        # Both are NaN - consider it valid
        if pd.isna(total_col_sum) and pd.isna(total_col_total):
            is_valid = True
        # One is NaN but other isn't - invalid
        elif pd.isna(total_col_sum) or pd.isna(total_col_total):
            is_valid = False
            self.logger.error(
                f"Total column total mismatch in '{total_column}': "
                f"Expected {total_col_total}, Computed {total_col_sum}"
            )
        # Both are numbers - check with tolerance
        else:
            is_valid = abs(total_col_sum - total_col_total) <= self.tolerance
            if not is_valid:
                self.logger.error(
                    f"Total column total mismatch in '{total_column}': "
                    f"Expected {total_col_total}, Computed {total_col_sum}, "
                    f"Difference: {abs(total_col_sum - total_col_total)}"
                )
        
        return is_valid
    
    def identify_columns_by_pattern(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Identify columns in the DataFrame that match predefined patterns.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with column categories and their corresponding column names
        """
        column_categories = {
            "balance_columns": [],
            "current_columns": [],
            "period1_columns": [],
            "period2_columns": [],
            "period3_columns": [],
            "period4_columns": [],
            "older_columns": [],
            "credit_limit_columns": []
        }
        
        # For each column in the DataFrame
        for col in df.columns:
            col_str = str(col).strip()
            
            # Check Balance column patterns
            for pattern in self.COLUMN_MAPPINGS["BALANCE_PATTERNS"]:
                if re.search(pattern, col_str, re.IGNORECASE):
                    column_categories["balance_columns"].append(col)
                    break
            
            # Check Current column patterns
            for pattern in self.COLUMN_MAPPINGS["CURRENT_PATTERNS"]:
                if re.search(pattern, col_str, re.IGNORECASE):
                    column_categories["current_columns"].append(col)
                    break
            
            # Check Period 1 column patterns
            for pattern in self.COLUMN_MAPPINGS["PERIOD1_PATTERNS"]:
                if re.search(pattern, col_str, re.IGNORECASE):
                    column_categories["period1_columns"].append(col)
                    break
            
            # Check Period 2 column patterns
            for pattern in self.COLUMN_MAPPINGS["PERIOD2_PATTERNS"]:
                if re.search(pattern, col_str, re.IGNORECASE):
                    column_categories["period2_columns"].append(col)
                    break
            
            # Check Period 3 column patterns
            for pattern in self.COLUMN_MAPPINGS["PERIOD3_PATTERNS"]:
                if re.search(pattern, col_str, re.IGNORECASE):
                    column_categories["period3_columns"].append(col)
                    break
            
            # Check Period 4 column patterns
            for pattern in self.COLUMN_MAPPINGS["PERIOD4_PATTERNS"]:
                if re.search(pattern, col_str, re.IGNORECASE):
                    column_categories["period4_columns"].append(col)
                    break
            
            # Check Older column patterns
            for pattern in self.COLUMN_MAPPINGS["OLDER_PATTERNS"]:
                if re.search(pattern, col_str, re.IGNORECASE):
                    column_categories["older_columns"].append(col)
                    break
            
            # Check Credit Limit column patterns
            for pattern in self.COLUMN_MAPPINGS["CREDIT_LIMIT_PATTERNS"]:
                if re.search(pattern, col_str, re.IGNORECASE):
                    column_categories["credit_limit_columns"].append(col)
                    break
        
        return column_categories
    
    def validate_csv(
        self,
        input_file: str,
        balance_columns: Optional[List[str]] = None,
        total_column: Optional[str] = None,
        output_file: Optional[str] = None,
        use_column_mapping: bool = True
    ) -> pd.DataFrame:
        try:
            self.logger.info(f"Starting validation of '{input_file}'")
            
            # Load data from CSV file (keep original data types)
            df = pd.read_csv(input_file, dtype=str)
            self.logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # If use_column_mapping is True, identify columns based on patterns
            if use_column_mapping:
                column_categories = self.identify_columns_by_pattern(df)
                
                # Find the total column (Balance)
                if not total_column:
                    balance_cols = column_categories["balance_columns"]
                    if balance_cols:
                        total_column = balance_cols[0]
                        self.logger.info(f"Automatically identified total column: '{total_column}'")
                    else:
                        self.logger.error("No balance column found in the CSV")
                        raise ValueError("No balance column found in the CSV")
                
                # Combine all period columns for row-wise validation
                if not balance_columns:
                    balance_columns = []
                    for key in ["current_columns", "period1_columns", "period2_columns", 
                                "period3_columns", "period4_columns", "older_columns"]:
                        balance_columns.extend(column_categories[key])
                    
                    if not balance_columns:
                        self.logger.error("No period columns found in the CSV")
                        raise ValueError("No period columns found in the CSV")
                    
                    self.logger.info(f"Automatically identified {len(balance_columns)} period columns")
            
            # Check if all required columns exist
            all_columns = balance_columns + [total_column]
            missing_columns = [col for col in all_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"Missing columns in CSV: {missing_columns}")
                raise ValueError(f"Missing columns in CSV: {missing_columns}")
            
            # Convert numeric columns
            numeric_columns = balance_columns + [total_column]
            numeric_df = self.get_numeric_dataframe(df, numeric_columns)
            
            # Find total row
            is_total_row, total_row_index = self.find_total_row(df, numeric_columns)
            
            # Create a copy of the original dataframe for adding validation results
            result_df = df.copy()
            
            # Validate column totals if total row exists
            column_validation_results = {}
            total_column_total_valid = None
            
            if is_total_row:
                column_validation_results = self.validate_column_totals(
                    numeric_df, balance_columns, total_row_index
                )
                
                # Validate the total column's total value
                total_column_total_valid = self.validate_total_column_total(
                    numeric_df, total_column, total_row_index
                )
                
                # Add validation results to result_df (not the original df)
                validation_row = pd.Series(index=df.columns)
                validation_row[df.columns[0]] = "Column_validationcheck"
                
                for col in balance_columns:
                    validation_row[col] = str(column_validation_results.get(col, "N/A"))
                
                # Add the total column validation result
                validation_row[total_column] = str(total_column_total_valid)
                
                # Add validation row after the total row or at the end
                result_df = pd.concat([result_df, pd.DataFrame([validation_row])], ignore_index=True)
            else:
                self.logger.warning("No total row found. Skipping column total validation.")
            
            # Validate row-wise balances
            row_validation_results = self.validate_row_balances(
                numeric_df, balance_columns, total_column, total_row_index
            )
            
            # Add row validation results as a new column - ONLY for the original rows
            # This ensures lengths match
            if is_total_row is False and total_row_index is None:
                result_df["Row_validationcheck"] = row_validation_results
            else:
                result_df["Row_validationcheck"] = row_validation_results + [None]  # Add None for the validation row
            
            # Calculate overall validation status
            column_valid = all(val for val in column_validation_results.values() if val is not None)
            row_valid = all(val for val in row_validation_results if val is not None)
            
            # Include total column total validation in overall result
            if total_column_total_valid is not None:
                overall_valid = column_valid and row_valid and total_column_total_valid
                self.logger.info(f"Total column total validation: {total_column_total_valid}")
            else:
                overall_valid = column_valid and row_valid
            
            self.logger.info(f"Validation complete - Overall valid: {overall_valid}")
            self.logger.info(f"Column validation: {column_valid}")
            self.logger.info(f"Row validation: {row_valid}")
            
            # Save the validated DataFrame to output file if specified
            if output_file:
                output_dir = os.path.dirname(output_file)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                
                result_df.to_csv(output_file, index=False)
                self.logger.info(f"Saved validated results to '{output_file}'")
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Error in validation: {str(e)}", exc_info=True)
            raise
    
    def validate_csvs_in_folder(self, input_folder: str, output_folder: str = None) -> None:
        """
        Validate all CSV files in the input folder and save results to the output folder.
        
        Args:
            input_folder: Path to the folder containing CSV files
            output_folder: Path to the folder to save validated results (default: input_folder/ValidatedOutputs)
        """
        try:
            # Create output folder if not specified
            if not output_folder:
                output_folder = os.path.join(input_folder, "ValidatedOutputs")
            
            # Create output folder if it doesn't exist
            if not os.path.exists(output_folder):
                os.makedirs(output_folder)
                self.logger.info(f"Created output folder: '{output_folder}'")
            
            # Find all CSV files in the input folder
            csv_files = glob.glob(os.path.join(input_folder, "*.csv"))
            self.logger.info(f"Found {len(csv_files)} CSV files in '{input_folder}'")
            
            if not csv_files:
                self.logger.warning(f"No CSV files found in '{input_folder}'")
                return
            
            # Validate each CSV file
            for csv_file in csv_files:
                filename = os.path.basename(csv_file)
                output_file = os.path.join(output_folder, filename)
                
                try:
                    self.validate_csv(
                        input_file=csv_file,
                        output_file=output_file,
                        use_column_mapping=True
                    )
                    self.logger.info(f"Successfully validated '{filename}'")
                except Exception as e:
                    self.logger.error(f"Error validating '{filename}': {str(e)}")
        
        except Exception as e:
            self.logger.error(f"Error in validate_csvs_in_folder: {str(e)}", exc_info=True)
            raise


def main():
    """Main function to run the CSV validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate CSV with numeric data and totals')
    parser.add_argument('--input-file', '-i', help='Path to input CSV file')
    parser.add_argument('--input-folder', '-f', help='Path to folder containing CSV files')
    parser.add_argument('--output-folder', '-o', help='Path to folder to save validated results')
    parser.add_argument('--balance-columns', '-b', nargs='+', help='Columns to sum for row validation')
    parser.add_argument('--total-column', '-t', help='Column containing the total amounts')
    parser.add_argument('--log-file', '-l', default="data_validation.log", help='Path to log file')
    parser.add_argument('--tolerance', type=float, default=0.01, help='Tolerance for floating point comparison')
    parser.add_argument('--use-column-mapping', '-m', action='store_true', 
                        help='Use column mapping to identify columns')
    
    args = parser.parse_args()
    
    validator = CSVValidator(log_file=args.log_file, tolerance=args.tolerance)
    
    if args.input_folder:
        validator.validate_csvs_in_folder(
            input_folder=args.input_folder,
            output_folder=args.output_folder
        )
        print(f"Validation of all CSV files completed. Results saved to '{args.output_folder or os.path.join(args.input_folder, 'ValidatedOutputs')}'")
    elif args.input_file:
        output_file = None
        if args.output_folder:
            output_file = os.path.join(args.output_folder, os.path.basename(args.input_file))
        
        validator.validate_csv(
            input_file=args.input_file,
            balance_columns=args.balance_columns,
            total_column=args.total_column,
            output_file=output_file,
            use_column_mapping=args.use_column_mapping
        )
        print(f"Validation completed. Check '{output_file or 'validated_output.csv'}' for results and '{args.log_file}' for logs.")
    else:
        print("Please provide either --input-file or --input-folder")
        parser.print_help()


# Example usage (when script is run directly)
if __name__ == "__main__":
    # If no command line arguments are provided, use these defaults
    import sys
    if len(sys.argv) == 1:
        # Use default folder path
        input_folder = r"C:\Users\singret\Downloads\OCR_validator"
        
        validator = CSVValidator()
        validator.validate_csvs_in_folder(input_folder=input_folder)
    else:
        main()