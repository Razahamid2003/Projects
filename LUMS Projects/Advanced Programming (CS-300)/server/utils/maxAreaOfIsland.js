
export function maxAreaOfIsland(grid) {
    if (!grid || grid.length === 0) {
      return 0;
    }
  
    const rows = grid.length;
    const cols = grid[0].length;
  
    function dfs(row, col, area) {
      if (
        row < 0 ||
        row >= rows ||
        col < 0 ||
        col >= cols ||
        grid[row][col] === 0
      ) {
        return;
      }
      grid[row][col] = 0;
      area.value += 1;
  
      dfs(row + 1, col, area);
      dfs(row - 1, col, area);
      dfs(row, col + 1, area);
      dfs(row, col - 1, area);
    }
  
    let maxArea = 0;
    for (let row = 0; row < rows; row++) {
      for (let col = 0; col < cols; col++) {
        if (grid[row][col] === 1) {
          const area = { value: 0 };
          dfs(row, col, area);
          if (area.value > maxArea) {
            maxArea = area.value;
          }
        }
      }
    }
  
    return maxArea;
  }
  