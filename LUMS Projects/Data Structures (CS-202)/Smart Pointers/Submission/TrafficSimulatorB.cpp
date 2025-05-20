#include <iostream>
#include <cstdlib>
#include "queue.cpp"
using namespace std;

// Car class representing each car at the junction
class Car {
    public:
        Car()//O(3)
        {
            id = 0;
            arrivalTime = 0;
            departureTime = 0;
        }

        Car(int i)//O(2)
        {
            id = i;
            departureTime = 0;
        }

        void setDetails(int i, int at)//O(2)
        {
            id = i;
            arrivalTime = at;
        }

        void setDeparture(int d)//O(1)
        {
            departureTime = d;
        }

        int getGlobal()//O(1)
        {
            return globalTime;
        }

        int getID()//O(1)
        {
            return id;
        }

        int getHighestID()//O(1)
        {
            return highest_ID;
        }

        int getArrival()//O(1)
        {
            return arrivalTime;
        }

        void incTime()//O(2)
        {
            globalTime++;
        }

        void incID()//O(2)
        {
           highest_ID++;
        }

    private:
        int id;
        int arrivalTime;
        int departureTime;
        static int globalTime;
        static int highest_ID;
};

int Car::highest_ID = 0;
int Car::globalTime = 0;

// TrafficSignal class representing each side of the junction
class TrafficSignal 
{
    public:
        TrafficSignal()
        {

        }

        TrafficSignal(string di) : direction(di)//O(5)
        {
            direction = di;
            Queue<Car> queue;
            totalTime = 0;
            queue_length = 0;
        }

        int getQueueLength()//O(1)
        {
            return queue_length;
        }

        void addCar(Car car)//O(10)
        {
            car.incTime();
            car.incID();
            car.setDetails(car.getHighestID(), car.getGlobal());
            cout<<"Car ID "<<car.getHighestID();
            queue.enqueue(car);
            queue_length++;
        }

        int moveCar()//O(27)
        {
            Car moved = queue.dequeue();
            moved.incTime();
            moved.setDeparture(moved.getGlobal());
            totalTime += (moved.getGlobal() - moved.getArrival());
            queue_length--;
            return moved.getID();
        }

        int carsDeparted()//O(1)
        {
            return cars_departed;
        }

        void resetCarsDeparted()//O(1)
        {
            cars_departed = 0;
        }

        void runSignal(int st)//O(n)
        {
            cars_departed = 0;
            random_queued = 0;
            totalTime = 0;
            for (int i = 0; i < st; i++)
            {
                if (queue.length() == 0)
                {
                    random_queued++;
                    cout<<"Signal empty.\n";
                }
                else
                {
                    int x = moveCar();
                    cars_departed++;
                    Car new_car;
                    cout<<"Car "<<x<<" crossed the signal at time "<<new_car.getGlobal()<<".\n";
                }
            }
        }

        void setDirection(string s)//O(1)
        {
            direction = s;
            queue_length = 0;
        }

        int getTotal()//O(1)
        {
            return totalTime;
        }

        void resetTotalTime()//O(1)
        {
            totalTime = 0;
        }
        
        int getRandomEnqueued()//O(1)
        {
            return random_queued;
        }

        Queue<Car> getQueue()//O(1)
        {
            return queue;
        }

        double getAvg()//O(5)
        {
            if (totalTime == 0 || cars_departed == 0)
            {
                return 0;
            }
            return static_cast<double>(totalTime)/static_cast<double>(cars_departed);
        }

    private:
        string direction;
        int queue_length;
        Queue<Car> queue;
        int totalTime;
        int cars_departed;
        int random_queued;
};

// Junction class representing the entire traffic signal system
class Junction 
{
    public:
    Junction()//O(4)
    {
        n.setDirection("North");
        s.setDirection("South");
        e.setDirection("East");
        w.setDirection("West");
    }

    Junction(int st)//O(5)
    {
        signalTime = st;
        n.setDirection("North");
        s.setDirection("South");
        e.setDirection("East");
        w.setDirection("West");
    }

    void addCar(Car car)//O(19)
    {
        int num = rand() % 4;

        switch (num)
        {
        case 0:
            n.addCar(car);
            cout<<" has queued at NORTH signal.\n";
            break;
        
        case 1:
            s.addCar(car);
            cout<<" has queued at SOUTH signal.\n";
            break;
        
        case 2:
            e.addCar(car);
            cout<<" has queued at EAST signal.\n";
            break;
        
        case 3:
            w.addCar(car);
            cout<<" has queued at WEST signal.\n";
            break;
        
        default:
            break;
        }
    }

    void runSimulation()//O(infinity). Considering it will run one cycle in a junction O(n)
    {
        bool end = false;
        int x[4] = {signalTime, signalTime, signalTime, signalTime};
        double avgs[4] = {0, 0, 0, 0};
        double k = 0.6;
        int prev_avg = 0;
        while (end == false)
        {
            int total_time = 0;
            int carsgone = 0;

            for (int i = 0; i < 4; i++)
            {
                switch (i)
                {
                case 0:
                    cout<<"Signal for North is OPEN for "<<x[i]<<" seconds.\n-----------------------------------------\n";
                    n.runSignal(x[i]);
                    total_time += n.getTotal();
                    carsgone += n.carsDeparted();
                    for (int i = 0; i <  (n.carsDeparted() + n.getRandomEnqueued()); i++)
                    {
                        Car temp_car;
                        addCar(temp_car);
                    }
                    avgs[i] =  n.getAvg();
                    
                    x[i] = x[i] - (n.getAvg()/n.getQueueLength());
                    cout<<"-----------------------------------------\n\n";
                    break;
                case 1:
                    cout<<"Signal for East is OPEN for "<<x[i]<<" seconds.\n-----------------------------------------\n";
                    e.runSignal(x[i]);
                    total_time += e.getTotal();
                    carsgone += e.carsDeparted();
                    for (int i = 0; i <  (e.carsDeparted() + e.getRandomEnqueued()); i++)
                    {
                        Car temp_car;
                        addCar(temp_car);
                    }
                    avgs[i] =  e.getAvg();
                    x[i] = x[i] - (e.getAvg()/e.getQueueLength());
                    cout<<"-----------------------------------------\n\n";
                    break;
                case 2:
                    cout<<"Signal for South is OPEN for "<<x[i]<<" seconds.\n-----------------------------------------\n";
                    s.runSignal(x[i]);
                    total_time += s.getTotal();
                    carsgone += s.carsDeparted();
                    for (int i = 0; i <  (s.carsDeparted() + s.getRandomEnqueued()); i++)
                    {
                        Car temp_car;
                        addCar(temp_car);
                    }
                    avgs[i] =  s.getAvg();
                    x[i] = x[i] - (s.getAvg()/s.getQueueLength());
                    cout<<"-----------------------------------------\n\n";
                    break;
                case 3:
                    cout<<"Signal for West is OPEN for "<<x[i]<<" seconds.\n-----------------------------------------\n";
                    w.runSignal(x[i]);
                    total_time += w.getTotal();
                    carsgone += w.carsDeparted();
                    for (int i = 0; i <  (w.carsDeparted() + w.getRandomEnqueued()); i++)
                    {
                        Car temp_car;
                        addCar(temp_car);
                    }
                    avgs[i] =  w.getAvg();
                    x[i] = x[i] - (w.getAvg()/w.getQueueLength());
                    cout<<"-----------------------------------------\n\n";
                    break;
                
                default:
                    break;
                }
            }
            cout<<"---------------------------------\nAverage Wait time for this cycle: "<<total_time/carsgone<<" seconds\n---------------------------------\n\n\n";

            x[0] = n.getQueueLength() - 0.005*(s.getAvg() + w.getAvg() + e.getAvg());
            x[1] = e.getQueueLength() - 0.005*(s.getAvg() + w.getAvg() + n.getAvg());
            x[2] = s.getQueueLength() - 0.005*(n.getAvg() + w.getAvg() + e.getAvg());
            x[3] = w.getQueueLength() - 0.005*(s.getAvg() + n.getAvg() + e.getAvg());

            n.resetCarsDeparted();
            n.resetTotalTime();
            s.resetCarsDeparted();
            s.resetTotalTime();
            e.resetCarsDeparted();
            e.resetTotalTime();
            w.resetCarsDeparted();
            w.resetTotalTime();
        }
    }

private:
    int signalTime;
    TrafficSignal n;
    TrafficSignal s;
    TrafficSignal e;
    TrafficSignal w;
};

int main() 
{
    cout<<"Enter a random Number: ";
    int x;
    cin>>x;
    cout<<"\n";
    srand(x);
    // Set the signal time (adjust as needed for part A)
    int signalTime = 10;

    // Create a junction with the given signal time
    Junction junction(signalTime);

    for (int i = 0; i < 20; i++)
    {
        Car temp_car;
        junction.addCar(temp_car);
    }

    // Run the simulation
    junction.runSimulation();

    return 0;
}