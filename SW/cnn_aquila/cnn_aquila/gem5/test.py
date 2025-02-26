import argparse
from m5.defines import buildEnv


parser = argparse.ArgumentParser()

parser.add_argument("-n", "--num-cpus", type=int, default=1)

parser.add_argument(
    "--mem-size",
    action="store",
    type=str,
    default="2GB",
    help="Specify the physical memory size (single memory, default:512MB)",
)

parser.add_argument(
    "--l1d_size",
    type=str,
    default="64kB",
    help="Specify the L1 data caches size (default: 64kB)",
)
parser.add_argument(
    "--l1i_size", 
    type=str, 
    default="16kB",
    help="Specify the L1 instruction caches size (default: 16kB)",
)
parser.add_argument(
    "--l2_size", 
    type=str, 
    default="256kB",
    help="Specify the L2 cache size (default: 256kB)",
)

parser.add_argument(
    "--l1d_assoc", 
    type=int, 
    default=2,
    help="Specify the L1 data caches associativity (default: 2)",
)
parser.add_argument(
    "--l1i_assoc", 
    type=int, 
    default=2,
    help="Specify the L1 instruction caches associativity (default: 2)",
)
parser.add_argument(
    "--l2_assoc", 
    type=int, 
    default=8,
    help="Specify the L2 cache associativity (default: 8)",
)

parser.add_argument(
    "--num-l2-banks",
    type=int,
    default=1,
    help="Specify the number of L2 cache banks (default: 1)"
)

parser.add_argument(
    "--ruby",
    action="store_true",
    help="Enable ruby memory syetem"
)
parser.add_argument(
    "--network-type",
    default="Simple",
    choices=["Simple", "Garnet"],
    help="Specify the Network on Chip (NoC) type (default: Simple)"
)
parser.add_argument(
    "--topology",
    default="Pt2Pt",
    choices=["Pt2Pt", "Mesh_XY", "Crossbar"],
    help="Specify the topology for the NoC on ruby memory system (default: Pt2Pt)"
)
parser.add_argument(
    "--cluster",
    action="store_true",
    help="Enable cluster topology, this is currently support ruby GarnetClusterCrossbar"
)

parser.add_argument(
    "--num-clusters",
    type=int,
    default=1,
    help="Specify the number of clusters when cluster topology is enabled (default: 1)"
)
parser.add_argument(
    "--mesh-rows",
    type=int,
    default=1,
    help="Specify the number of mesh rows when topology is specified to Mesh_XY (default: 1)"
)

parser.add_argument(
    "--model",
    default="ultranet",
    choices=["ultranet", "resnet18", "resnet50", "alexnet"],
    help="Specify the CNN model (default: ultranet)"
)
parser.add_argument(
    "--input-file",
    default="ultranet_2.ppm",
    choices=["Scottish_deerhound.bin", "Samoyed.bin", "ultranet_2.ppm", "ultranet_3.ppm", "ultranet_4.ppm", "ultranet_5.ppm"],
    help="[Scottish_deerhound, Samoyed] is for [resnet18, resnet50, alexnet]. And [ultranet_1~5] is for ultranet only. (default: ultranet_2.ppm)"
)

args = parser.parse_args()

ultranet_input = ["ultranet_2.ppm", "ultranet_3.ppm", "ultranet_4.ppm", "ultranet_5.ppm"]
if (args.model == "ultranet") ^ (args.input_file in ultranet_input):
    raise ValueError("Model type and input not matched")


from gem5.components.realtek.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.processors.simple_processor import SimpleProcessor
from gem5.components.processors.cpu_types import CPUTypes
from gem5.components.realtek.boards.simple_board import SimpleBoard
from gem5.isas import ISA
from gem5.resources.resource import BinaryResource
from gem5.simulate.simulator import Simulator
import m5
import time

if not args.ruby:
    exec("""
from gem5.components.cachehierarchies.classic.private_l1_shared_l2_cache_hierarchy \
import PrivateL1SharedL2CacheHierarchy as CacaheHierarchy
""")
    exec(f"""
cache_hierarchy = CacaheHierarchy(
    l1i_size="{args.l1i_size}", 
    l1i_assoc={args.l1i_assoc}, 
    l1d_size="{args.l1d_size}",
    l1d_assoc={args.l1d_assoc}, 
    l2_size="{args.l2_size}",
    l2_assoc={args.l2_assoc}
)
""")
else: # if --ruby specified
    # for now, MI_example does not support cluster topology
    if buildEnv['PROTOCOL'] == 'MI_example':
        args.cluster = False


    topo_arg = ""
    if args.cluster:
        args.topology = "Crossbar"
        args.network_type = "Garnet"
        topo_arg = f"num_clusters={args.num_clusters}, num_l2_banks={args.num_l2_banks}"
    elif args.topology == "Mesh_XY":
        topo_arg = f"num_cpus={args.num_cpus}, mesh_rows={args.mesh_rows}"
    # import network topology
    exec(f"""
from gem5.components.realtek.cachehierarchies.ruby.topologies.{args.network_type.lower()}\
.{args.network_type.lower()}_{"cluster_" if args.cluster else ""}{args.topology.lower()} \
import {args.network_type}{"Cluster" if args.cluster else ""}{args.topology} \
as topo

network = topo({topo_arg})
""")
    if buildEnv['PROTOCOL'] == "MESI_Two_Level":
        exec(f"""
from gem5.components.realtek.cachehierarchies.ruby.mesi_two_level_{"cluster_" if args.cluster else ""}cache_hierarchy \
import MESITwoLevel{"Cluster" if args.cluster else ""}CacheHierarchy as CacaheHierarchy

cache_hierarchy = CacaheHierarchy(
    l1i_size="{args.l1i_size}", 
    l1i_assoc={args.l1i_assoc}, 
    l1d_size="{args.l1d_size}",
    l1d_assoc={args.l1d_assoc}, 
    l2_size="{args.l2_size}",
    l2_assoc={args.l2_assoc}, 
    num_l2_banks={args.num_l2_banks},
    network=network,
    {f"num_clusters={args.num_clusters}" if args.cluster else ""}
)
""")
    elif buildEnv['PROTOCOL'] == 'MI_example':
        exec(f"""
from gem5.components.realtek.cachehierarchies.ruby.mi_example_cache_hierarchy import MIExampleCacheHierarchy as CacaheHierarchy
             
cache_hierarchy = CacaheHierarchy(
    size="{args.l1d_size}",
    assoc={args.l1d_assoc},
    network=network
)
""")
    else:
        raise TypeError("Unsupported protocol")

memory = SingleChannelDDR3_1600(args.mem_size, num_dirs=1)
processor = SimpleProcessor(isa=ISA.RISCV, cpu_type=CPUTypes.TIMING, num_cores=args.num_cpus)
board = SimpleBoard(
    clk_freq="1GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy,
)



# Set the workload.
binary = BinaryResource(f"./{args.model}")
board.set_multiple_se_binary_workload(
    binary = [binary]*args.num_cpus,
    arguments = [[args.num_cpus, i] for i in range(args.num_cpus)]
)

board.workload.extras = [f"./input_file/{args.input_file}", f"./weights/{args.model}_Weight", "./gem5/imagenet_classes.txt"]
board.workload.extras_addrs = [0x0000000010000000, 0x0000000020000000, 0x0000000041000000]
from m5.objects import Addr

def SimulatorMap(instance):
    for core in instance._board.get_processor().get_cores():
        core.core.workload[0].map(Addr(0xC000000000000000), Addr(0x0000000040000000), 2048*20, True) # network pointer
        core.core.workload[0].map(Addr(0xC100000000000000), Addr(0x0000000010000000), 2048*1024*256, True) # layers pointer
        core.core.workload[0].map(Addr(0xC200000000000000), Addr(0x0000000020000000), 2048*1024*128, True) # weight pointer
        core.core.workload[0].map(Addr(0xC300000000000000), Addr(0x0000000041000000), 2048*32, True) # class name pointer
            
simulator = Simulator(board=board)


print("-------------------------------------------------------------------------------------")
if not args.ruby:
    print(f"Simulation on {args.num_cpus} cores with Classic memory system starts")
else:
    print(f"Simulation on {args.num_cpus} cores with Ruby {buildEnv['PROTOCOL']} {args.network_type} Network {'Cluster' if args.cluster else ''}{args.topology}{f'({args.mesh_rows}x{args.num_cpus//args.mesh_rows})' if args.topology == 'Mesh_XY' else ''} starts")
print(f"Inferencing {args.input_file} using {args.model} model")
print("-------------------------------------------------------------------------------------")

start = time.time()
simulator.run(func=SimulatorMap)
end = time.time()


print("-------------------------------------------------------------------------------------")
if not args.ruby:
    print(f"Simulation on {args.num_cpus} cores with Classic memory system ends")
else:
    print(f"Simulation on {args.num_cpus} cores with Ruby {buildEnv['PROTOCOL']} {args.network_type} Network {'Cluster' if args.cluster else ''}{args.topology}{f'({args.mesh_rows}x{args.num_cpus//args.mesh_rows})' if args.topology == 'Mesh_XY' else ''} ends")


print("-------------------------------------------------------------------------------------")
print(time.strftime('%d %H:%M:%S', time.gmtime(end - start)))


print("Exiting @ tick", m5.curTick(), "because", simulator._last_exit_event.getCause())