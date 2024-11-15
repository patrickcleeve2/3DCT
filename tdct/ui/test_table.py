from PyQt5.QtCore import Qt, QAbstractTableModel, pyqtSignal, QModelIndex
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTableView, 
                           QHeaderView, QVBoxLayout, QWidget)
import pandas as pd
import numpy as np

class ReorderableTableView(QTableView):
    """Custom TableView that enables drag and drop for rows only"""
    def __init__(self):
        super().__init__()
        # Enable drag and drop
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QTableView.InternalMove)
        self.setDragDropOverwriteMode(False)
        
        # Enable selection
        self.setSelectionBehavior(QTableView.SelectRows)
        self.setSelectionMode(QTableView.SingleSelection)
        
        # Set up headers
        self.horizontalHeader().setSectionsMovable(False)  # Disable column moving
        self.verticalHeader().setSectionsMovable(True)
        self.verticalHeader().setDragEnabled(True)
        self.verticalHeader().setDragDropMode(QHeaderView.InternalMove)
        
        # Make the vertical header (row numbers) visible and styled
        self.verticalHeader().setVisible(True)
        self.verticalHeader().setDefaultAlignment(Qt.AlignRight | Qt.AlignVCenter)
        
        # Stretch columns to fill the widget
        self.horizontalHeader().setStretchLastSection(True)

class PandasTableModel(QAbstractTableModel):
    """Custom table model to sync PyQt5 TableView with pandas DataFrame"""
    dataChanged = pyqtSignal(pd.DataFrame)
    
    def __init__(self, df=pd.DataFrame()):
        super().__init__()
        self._df = df.copy()
        self._df.index = range(1, len(df) + 1)  # 1-based index
        self.row_order = list(range(len(df)))

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid():
            return None
            
        if role == Qt.DisplayRole:
            # Map the view index to the actual DataFrame index
            row = self.row_order[index.row()]
            value = self._df.iloc[row, index.column()]
            
            if pd.isna(value):
                return ''
            if isinstance(value, (float, np.floating)):
                return f"{value:.2f}"
            return str(value)
            
        return None

    def setData(self, index, value, role=Qt.EditRole):
        if role == Qt.EditRole:
            try:
                row = self.row_order[index.row()]
                col_name = self._df.columns[index.column()]
                current_dtype = self._df[col_name].dtype
                
                if pd.api.types.is_numeric_dtype(current_dtype):
                    value = pd.to_numeric(value)
                
                self._df.iloc[row, index.column()] = value
                self.dataChanged.emit(self._df)
                return True
            except (ValueError, TypeError):
                return False
        return False

    def rowCount(self, parent=None):
        return len(self._df)

    def columnCount(self, parent=None):
        return len(self._df.columns)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role == Qt.DisplayRole:
            if orientation == Qt.Horizontal:
                return str(self._df.columns[section])
            else:
                # Show the actual index value (1-based)
                return str(section + 1)
        elif role == Qt.TextAlignmentRole and orientation == Qt.Vertical:
            return Qt.AlignRight | Qt.AlignVCenter
        return None

    def flags(self, index):
        return Qt.ItemIsEditable | Qt.ItemIsEnabled | Qt.ItemIsSelectable | \
               Qt.ItemIsDragEnabled | Qt.ItemIsDropEnabled

    def update_df(self, new_df):
        """Update the model's DataFrame and refresh the view"""
        self.layoutAboutToBeChanged.emit()
        self._df = new_df.copy()
        self._df.index = range(1, len(new_df) + 1)  # Reset to 1-based index
        self.row_order = list(range(len(new_df)))
        self.layoutChanged.emit()

    def moveRow(self, source_row, dest_row):
        """Handle row reordering"""
        if source_row == dest_row:
            return

        self.layoutAboutToBeChanged.emit()
        
        # Update the row order
        moving_row = self.row_order.pop(source_row)
        self.row_order.insert(dest_row, moving_row)
        
        # Create new DataFrame with reordered rows
        temp_df = self._df.iloc[self.row_order].copy()
        
        # Reset the index to be 1-based consecutive numbers
        temp_df.index = range(1, len(temp_df) + 1)
        
        # Update the DataFrame and row order
        self._df = temp_df
        self.row_order = list(range(len(self._df)))
        
        # Emit signals to update the view
        self.layoutChanged.emit()
        self.dataChanged.emit(self._df)

    def get_data(self):
        """Return the current state of the DataFrame"""
        return self._df.copy()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 400)
        
        # Create sample DataFrame
        self.df = pd.DataFrame({
            'Name': ['John', 'Alice', 'Bob', 'Carol'],
            'Age': [25, 30, 35, 28],
            'Salary': [50000.0, 65000.0, 75000.0, 60000.0],
            'Department': ['HR', 'Engineering', 'Sales', 'Marketing']
        })
        
        # Create table view and model
        self.table = QTableView()
        self.model = PandasTableModel(self.df)
        self.table.setModel(self.model)
        
        # Connect signals
        self.model.dataChanged.connect(self.on_data_changed)
        self.table.verticalHeader().sectionMoved.connect(self.on_row_moved)
        
        # Create central widget and layout
        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.addWidget(self.table)
        self.setCentralWidget(central_widget)
        
        # Set column widths
        header = self.table.horizontalHeader()
        for column in range(self.model.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.Interactive)

    def on_data_changed(self, df):
        print("DataFrame updated:")
        print(df)

    def on_row_moved(self, logical_index, old_visual_index, new_visual_index):
        self.model.moveRow(old_visual_index, new_visual_index)

if __name__ == '__main__':
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()