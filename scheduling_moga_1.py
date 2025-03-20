import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
from datetime import datetime, timedelta
import copy

# Reset DEAP creators if they exist
if 'FitnessMulti' in creator.__dict__:
    del creator.FitnessMulti
if 'Individual' in creator.__dict__:
    del creator.Individual

class ShopfloorScheduler:
    """Main class for shopfloor scheduling optimization"""
    
    def __init__(self):
        # Initialize DEAP genetic algorithm components
        creator.create("FitnessMulti", base.Fitness, weights=(-5.0, -1.0, -0.5, 1.0))
        creator.create("Individual", list, fitness=creator.FitnessMulti)
        self.toolbox = base.Toolbox()
        
    def load_data(self, excel_file):
        """Load data from Excel file with multiple sheets"""
        self.data = {
            'orders': pd.read_excel(excel_file, sheet_name='Manufacturing_Orders'),
            'products': pd.read_excel(excel_file, sheet_name='Products'),
            'operations': pd.read_excel(excel_file, sheet_name='Operations'),
            'machines': pd.read_excel(excel_file, sheet_name='Machines'),
            'machine_operations': pd.read_excel(excel_file, sheet_name='Machine_Operations'),
            'product_operations': pd.read_excel(excel_file, sheet_name='Product_Operations')
        }
        
        # Convert date columns
        self.data['orders']['required_date'] = pd.to_datetime(self.data['orders']['required_date'])
        
    def generate_initial_schedule(self):
        """Generate initial schedule based on earliest due date"""
        schedule = []
        orders = self.data['orders'].sort_values('required_date')
        
        for _, order in orders.iterrows():
            # Get operations for this product
            operations = self.get_product_operations(order['product_id'])
            # Calculate optimal batches
            batches = self.calculate_batches(order, operations)
            
            for batch in batches:
                for op in operations:
                    # Find best machine
                    machine = self.find_best_machine(op['operation_id'])
                    
                    schedule.append({
                        'order_id': order['order_id'],
                        'product_id': order['product_id'],
                        'operation_id': op['operation_id'],
                        'machine_id': machine,
                        'batch_size': batch['size'],
                        'sequence': op['sequence'],
                        'priority': batch['priority']
                    })
        
        return schedule
    
    def get_product_operations(self, product_id):
        """Get ordered list of operations for a product"""
        ops = self.data['product_operations'][
            self.data['product_operations']['product_id'] == product_id
        ].sort_values('sequence')
        
        return [
            {
                'operation_id': row['operation_id'],
                'sequence': row['sequence']
            }
            for _, row in ops.iterrows()
        ]
    
    def calculate_batches(self, order, operations):
        """Calculate optimal batch sizes for an order"""
        total_qty = order['total_quantity']
        product = self.data['products'][
            self.data['products']['product_id'] == order['product_id']
        ].iloc[0]
        
        min_batch = product['min_batch_size']
        max_batch = product['max_batch_size']
        
        batches = []
        remaining_qty = total_qty
        priority = 1
        
        while remaining_qty > 0:
            batch_size = min(max_batch, remaining_qty)
            if batch_size < min_batch:
                batch_size = remaining_qty
                
            batches.append({
                'size': batch_size,
                'priority': priority
            })
            
            remaining_qty -= batch_size
            priority += 1
            
        return batches
    
    def find_best_machine(self, operation_id):
        """Find best machine for an operation based on efficiency"""
        compatible_machines = self.data['machine_operations'][
            self.data['machine_operations']['operation_id'] == operation_id
        ]
        return compatible_machines.sort_values('speed_factor', ascending=False)['machine_id'].iloc[0]
    
    def create_schedule(self, individual):
        """Convert genetic algorithm individual to actual schedule"""
        schedule = []
        machine_schedules = {m: [] for m in self.data['machines']['machine_id']}
        current_time = pd.Timestamp('2024-04-01 08:00:00')
        
        # Sort operations by priority and sequence
        sorted_ops = sorted(individual, key=lambda x: (x['priority'], x['sequence']))
        
        for op in sorted_ops:
            machine_id = op['machine_id']
            
            # Calculate processing time
            base_time = float(self.data['operations'][
                self.data['operations']['operation_id'] == op['operation_id']
            ]['processing_time_per_unit_mins'].iloc[0])
            
            machine_factor = float(self.data['machine_operations'][
                (self.data['machine_operations']['machine_id'] == machine_id) &
                (self.data['machine_operations']['operation_id'] == op['operation_id'])
            ]['speed_factor'].iloc[0])
            
            processing_time = base_time * op['batch_size'] / machine_factor
            
            # Find earliest available time on machine
            if machine_schedules[machine_id]:
                start_time = max(
                    current_time,
                    machine_schedules[machine_id][-1]['end_time']
                )
            else:
                start_time = current_time
            
            # Calculate setup time
            setup_time = self.calculate_setup_time(machine_schedules[machine_id], op)
            
            # Calculate end time
            end_time = start_time + pd.Timedelta(minutes=processing_time + setup_time)
            
            schedule_entry = {
                'order_id': op['order_id'],
                'product_id': op['product_id'],
                'operation_id': op['operation_id'],
                'machine_id': machine_id,
                'batch_size': op['batch_size'],
                'sequence': op['sequence'],
                'start_time': start_time,
                'end_time': end_time,
                'setup_time': setup_time,
                'processing_time': processing_time
            }
            
            schedule.append(schedule_entry)
            machine_schedules[machine_id].append(schedule_entry)
        
        return pd.DataFrame(schedule)
    
    def calculate_setup_time(self, machine_schedule, next_op):
        """Calculate setup time between operations"""
        if not machine_schedule:
            return 0
        
        last_op = machine_schedule[-1]
        base_setup_time = float(self.data['operations'][
            self.data['operations']['operation_id'] == next_op['operation_id']
        ]['setup_time_mins'].iloc[0])
        
        # Additional setup time if different products
        if last_op['product_id'] != next_op['product_id']:
            base_setup_time *= 1.5
            
        return base_setup_time
    
    def evaluate_schedule(self, individual):
        """Evaluate schedule quality"""
        schedule = self.create_schedule(individual)
        
        # 1. Calculate delays
        order_completion = schedule.groupby('order_id')['end_time'].max()
        orders_with_completion = self.data['orders'].join(
            order_completion.to_frame('completion_time'),
            on='order_id'
        )
        
        total_delay = sum(
            max(0, (completion - required).total_seconds() / 3600)
            for completion, required in zip(
                orders_with_completion['completion_time'],
                orders_with_completion['required_date']
            )
        )
        
        # 2. Calculate makespan
        makespan = (
            schedule['end_time'].max() - 
            schedule['start_time'].min()
        ).total_seconds() / 3600
        
        # 3. Calculate setup time
        total_setup = schedule['setup_time'].sum()
        
        # 4. Calculate machine utilization
        total_time = makespan
        machine_utilization = schedule.groupby('machine_id').agg({
            'processing_time': 'sum',
            'setup_time': 'sum'
        })
        avg_utilization = (
            (machine_utilization['processing_time'] + machine_utilization['setup_time'])
            / (total_time * 60)  # Convert hours to minutes
        ).mean() * 100
        
        return (total_delay, makespan, total_setup, -avg_utilization)
    
    def setup_genetic_algorithm(self):
        """Setup genetic algorithm operators"""
        self.toolbox.register(
            "individual",
            tools.initIterate,
            creator.Individual,
            self.generate_initial_schedule
        )
        self.toolbox.register(
            "population",
            tools.initRepeat,
            list,
            self.toolbox.individual
        )
        self.toolbox.register("evaluate", self.evaluate_schedule)
        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)
        self.toolbox.register("select", tools.selNSGA2)

    def optimize(self, population_size=100, generations=50):
        """Run the optimization"""
        print("Starting optimization...")
        
        # Initialize population
        pop = self.toolbox.population(n=population_size)
        
        # Store initial best schedule
        initial_schedule = self.create_schedule(pop[0])
        initial_metrics = self.calculate_metrics(initial_schedule)
        
        # Run optimization
        final_pop = algorithms.eaMuPlusLambda(
            pop, self.toolbox,
            mu=population_size,
            lambda_=population_size,
            cxpb=0.7,  # Crossover probability
            mutpb=0.2,  # Mutation probability
            ngen=generations,
            verbose=True
        )
        
        # Get best solution
        best_individual = tools.selBest(final_pop[0], k=1)[0]
        final_schedule = self.create_schedule(best_individual)
        final_metrics = self.calculate_metrics(final_schedule)
        
        return final_schedule, initial_metrics, final_metrics
    
    def calculate_metrics(self, schedule):
        """Calculate comprehensive schedule metrics"""
        metrics = {}
        
        # Order fulfillment metrics
        order_completion = schedule.groupby('order_id')['end_time'].max()
        orders_with_completion = self.data['orders'].join(
            order_completion.to_frame('completion_time'),
            on='order_id'
        )
        
        metrics['on_time_delivery_rate'] = (
            sum(completion <= required
                for completion, required in zip(
                    orders_with_completion['completion_time'],
                    orders_with_completion['required_date']
                )
            ) / len(orders_with_completion) * 100
        )
        
        metrics['delayed_orders'] = sum(
            completion > required
            for completion, required in zip(
                orders_with_completion['completion_time'],
                orders_with_completion['required_date']
            )
        )
        
        # Efficiency metrics
        metrics['makespan_days'] = (
            schedule['end_time'].max() - 
            schedule['start_time'].min()
        ).total_seconds() / (24 * 3600)
        
        # Machine utilization
        total_time = metrics['makespan_days'] * 24 * 60  # Convert to minutes
        machine_utilization = schedule.groupby('machine_id').agg({
            'processing_time': 'sum',
            'setup_time': 'sum'
        })
        metrics['avg_machine_utilization'] = (
            (machine_utilization['processing_time'] + machine_utilization['setup_time'])
            / total_time
        ).mean() * 100
        
        return metrics

def main():
    """Main execution function"""
    # Initialize scheduler
    scheduler = ShopfloorScheduler()
    
    # Generate or load data
    scheduler.load_data('shopfloor_scheduling_data.xlsx')
    
    # Setup genetic algorithm
    scheduler.setup_genetic_algorithm()
    
    # Run optimization
    final_schedule, initial_metrics, final_metrics = scheduler.optimize(
        population_size=100,
        generations=50
    )
    
    # Save results
    final_schedule.to_excel('optimized_schedule.xlsx', index=False)
    
    # Create comparison report
    comparison = pd.DataFrame({
        'Metric': list(initial_metrics.keys()),
        'Before': list(initial_metrics.values()),
        'After': list(final_metrics.values()),
        'Improvement': [
            f"{((after - before) / before) * 100:.1f}%"
            for before, after in zip(
                initial_metrics.values(),
                final_metrics.values()
            )
        ]
    })
    
    comparison.to_excel('optimization_results.xlsx', index=False)
    print("\nOptimization complete! Check 'optimized_schedule.xlsx' and 'optimization_results.xlsx'")

if __name__ == "__main__":
    main()
