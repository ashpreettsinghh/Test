import pandas as pd
import logging
import os
from typing import List, Dict, Tuple, Optional, Union, Any
import re


class CSVValidator:
    """
    A class to validate CSV files with numeric data and total rows/columns.
    
    This validator checks:
    1. If a total row exists and matches the sum of other rows
    2. If the row-wise balance matches the specified total column
    3. If the total column's total matches the sum of other total column values
    
    The validator now supports column mapping to handle different header formats.
    """
    
    # Patterns to identify total rows in data
    TOTAL_ROW_PATTERNS = [
        r'total',
        r'totals',
        r'Total (LCY)',
        r'Grand Totals',
        r'ReportTotals',
    ]
    
    # Default column mappings for aging buckets
    DEFAULT_COLUMN_MAPPINGS = {
        'current': ['Current', '<30', '1-30', '0-30', '0 - 30', 'Current Period', 'Not Due'],
        'period1': ['1 30 days', '1-30 days', '1-30', '30-60', '31-60', '31 60 days', '30 - 60', '1-30 Days Past Due'],
        'period2': ['31 60 days', '61-90', '61 90 days', '60-90', '61-90', '60 - 90', '31-60 Days Past Due'],
        'period3': ['61 90 days', '91-120', '91 120 days', '90-120', '91-120', '90 - 120', '61-90 Days Past Due'],
        'period4': ['91 120 days', '121-150', '121 150 days', '120-150', '121-150', '120 - 150', '91-120 Days Past Due'],
        'period5': ['121 150 days', '151-180', '151 180 days', '150-180', '151-180', '150 - 180', '121-150 Days Past Due'],
        'period6': ['151 180 days', '181-210', '181 210 days', '180-210', '181-210', '180 - 210', '151-180 Days Past Due'],
        'older': ['Before 120 days', '> 180', '>180', '180+', '210+', 'Over 180', '181+ Days Past Due', 'Older'],
        'balance': ['Balance', 'Total', 'Amount', 'Outstanding', 'Total Amount', 'Balance Due', 'Outstanding Balance']
    }
    
    def __init__(
        self, 
        log_file: str = "data_validation.log", 
        log_level: int = logging.INFO,
        tolerance: float = 0.01,
        column_mappings: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the CSV validator.
        
        Args:
            log_file: Path to log file
            log_level: Logging level
            tolerance: Tolerance for floating point comparison
            column_mappings: Custom column mappings dict (key: column_type, value: list of possible headers)
        """
        self.setup_logging(log_file, log_level)
        self.tolerance = tolerance
        self.logger = logging.getLogger(__name__)
        
        # Initialize column mappings with defaults, then update with any custom mappings
        self.column_mappings = self.DEFAULT_COLUMN_MAPPINGS.copy()
        if column_mappings:
            for key, values in column_mappings.items():
                self.column_mappings[key] = values
    
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
        balance_columns: List[str],
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
        computed_totals = data_rows[balance_columns].sum()
        provided_totals = numeric_df.loc[total_row_index, balance_columns]
        
        for col in balance_columns:
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
    
    def map_columns(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Map actual column names in the dataframe to standardized column types.
        
        Args:
            df: DataFrame with headers to map
            
        Returns:
            Dictionary mapping column types to actual column names
        """
        column_mapping_result = {}
        
        # Iterate through each column type and its possible headers
        for column_type, possible_headers in self.column_mappings.items():
            # Check if any of the possible headers exist in the dataframe
            for header in possible_headers:
                if header in df.columns:
                    column_mapping_result[column_type] = header
                    self.logger.info(f"Mapped '{column_type}' to column '{header}'")
                    break
        
        # Log which column types were not found
        for column_type in self.column_mappings:
            if column_type not in column_mapping_result:
                self.logger.warning(f"Could not find a matching column for '{column_type}'")
        
        return column_mapping_result
    
    def get_balance_columns(self, mapped_columns: Dict[str, str]) -> List[str]:
        """
        Get the list of balance columns from the mapped columns.
        
        Args:
            mapped_columns: Dictionary mapping column types to actual column names
            
        Returns:
            List of actual column names that represent balance columns
        """
        # Balance columns are all aging bucket columns (excluding 'balance')
        balance_column_types = [
            'current', 'period1', 'period2', 'period3', 
            'period4', 'period5', 'period6', 'older'
        ]
        
        return [mapped_columns[col_type] for col_type in balance_column_types 
                if col_type in mapped_columns]
    
    def validate_csv(
        self,
        input_file: str,
        balance_columns: Optional[List[str]] = None,
        total_column: Optional[str] = None,
        output_file: Optional[str] = None,
        use_column_mapping: bool = False
    ) -> pd.DataFrame:
        try:
            self.logger.info(f"Starting validation of '{input_file}'")
            
            # Load data from CSV file (keep original data types)
            df = pd.read_csv(input_file, dtype=str)
            self.logger.info(f"Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            
            # If using column mapping, determine the columns to use
            mapped_columns = {}
            if use_column_mapping:
                self.logger.info("Using column mapping to identify aging columns and total column")
                mapped_columns = self.map_columns(df)
                
                # Get balance columns and total column from mapping if not explicitly provided
                if balance_columns is None:
                    balance_columns = self.get_balance_columns(mapped_columns)
                    self.logger.info(f"Using mapped balance columns: {balance_columns}")
                
                if total_column is None and 'balance' in mapped_columns:
                    total_column = mapped_columns['balance']
                    self.logger.info(f"Using mapped total column: {total_column}")
            
            # If balance_columns or total_column is still None, we need to raise an error
            if balance_columns is None:
                self.logger.error("No balance columns specified or found in mapping")
                raise ValueError("No balance columns specified or found in mapping")
            
            if total_column is None:
                self.logger.error("No total column specified or found in mapping")
                raise ValueError("No total column specified or found in mapping")
            
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
            if len(row_validation_results) < len(result_df):
                # Add None for any extra rows (like validation rows)
                row_validation_results = row_validation_results + [None] * (len(result_df) - len(row_validation_results))
            
            result_df["Row_validationcheck"] = row_validation_results
            
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
            

def main():
    """Main function to run the CSV validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Validate CSV with numeric data and totals')
    parser.add_argument('input_file', help='Path to input CSV file')
    parser.add_argument('--balance-columns', '-b', nargs='+',
                      help='Columns to sum for row validation')
    parser.add_argument('--total-column', '-t',
                      help='Column containing the total amounts')
    parser.add_argument('--output-file', '-o', default="validated_output.csv",
                      help='Path to output CSV file')
    parser.add_argument('--log-file', '-l', default="data_validation.log",
                      help='Path to log file')
    parser.add_argument('--tolerance', type=float, default=0.01,
                      help='Tolerance for floating point comparison')
    parser.add_argument('--use-mapping', '-m', action='store_true',
                      help='Use column mapping to automatically identify columns')
    parser.add_argument('--add-mapping', '-a', nargs=2, action='append', metavar=('column_type', 'header_name'),
                      help='Add a custom column mapping (can be used multiple times)')
    
    args = parser.parse_args()
    
    # Process custom column mappings
    custom_mappings = {}
    if args.add_mapping:
        for column_type, header_name in args.add_mapping:
            if column_type not in custom_mappings:
                custom_mappings[column_type] = []
            custom_mappings[column_type].append(header_name)
    
    validator = CSVValidator(
        log_file=args.log_file,
        tolerance=args.tolerance,
        column_mappings=custom_mappings if custom_mappings else None
    )
    
    validator.validate_csv(
        args.input_file,
        args.balance_columns,
        args.total_column,
        args.output_file,
        use_column_mapping=args.use_mapping
    )
    
    print(f"Validation completed. Check '{args.output_file}' for results and '{args.log_file}' for logs.")


# Example usage (when script is run directly)
if __name__ == "__main__":
    # If no command line arguments are provided, use these defaults
    import sys
    if len(sys.argv) == 1: #ensure header starts from first line in csv
        input_file = fr"C:\Users\singret\Downloads\OCR_validator\output_RBS Company C - Aged Accounts Receivable.pdf_extracted_tables_Table_1.csv"  # Replace with your default file path
        
        # You can either specify columns explicitly OR use column mapping
        use_column_mapping = True  # Set to True to use automatic column mapping
        
        # These are used only if use_column_mapping is False
        total_column_name = 'Balance'
        balance_columns = ['1 30 days', '31 60 days', '61 90 days', '91 120 days', 'Before 120 days']
        
        # Example of adding custom mappings to the defaults
        custom_mappings = {
            'current': ['<30', '1-30', '0-30', 'Current'],
            'period1': ['31-60', '30-60', '1-30 days'],
            'period2': ['61-90', '60-90', '31 60 days'],
            'period3': ['91-120', '90-120', '61 90 days'],
            'older': ['Before 120 days', '120+', 'Over 120'],
            'balance': ['Balance', 'Total', 'Amount Due']
        }
        
        validator = CSVValidator(column_mappings=custom_mappings)
        
        if use_column_mapping:
            validator.validate_csv(
                input_file=input_file,
                output_file="validated_output.csv",
                use_column_mapping=True
            )
        else:
            validator.validate_csv(
                input_file=input_file,
                balance_columns=balance_columns,
                total_column=total_column_name,
                output_file="validated_output.csv"
            )
    else:
        main()