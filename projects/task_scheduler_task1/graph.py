class DependencyGraph:
    def __init__(self):
        self.dependencies = {}  # task_id -> list of dependencies

    def add_dependency(self, task_id, dependency_id):
        if task_id not in self.dependencies:
            self.dependencies[task_id] = []
        self.dependencies[task_id].append(dependency_id)

    def find_order(self, task_ids) -> list:
        """Finds a valid execution order using topological sort."""
        visited = set()
        order = []
        for task_id in task_ids:
            if task_id not in visited:
                self._topo_sort(task_id, visited, order)
        return order

    def _topo_sort(self, task_id, visited, order):
        visited.add(task_id)
        for dep in self.dependencies.get(task_id, []):
            if dep not in visited:
                self._topo_sort(dep, visited, order)
        order.append(task_id)
