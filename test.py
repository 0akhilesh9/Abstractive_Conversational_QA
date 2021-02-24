from optparse import OptionParser

import utils as utils
import train as train

config_file = r"project.config"

if __name__ == "__main__":
    usageStr = """
               USAGE:      python test.py --conf <config_file_path>
               """
    parser = OptionParser(usageStr)

    parser.add_option("--conf", type=str, dest="config_file", default=config_file, help="Config File")
    parser.add_option("--abstract", dest="abstract", default=False, help="Make abstract generation", action="store_true")
    parser.add_option("--show_graphs", dest="show_graphs", default=False, help="Display graphs", action="store_true")
    parser.add_option("--pred_file", type=str, dest="pred_file", default="", help="Predictions file")
    parser.add_option("--actual_file", type=str, dest="actual_file", default="", help="Actual answer file")
    options, otherjunk = parser.parse_args()
    config_file = options.config_file
  
    utils.config.read_file(open(config_file))
    if not options.abstract:
        train.train_model()
    elif options.abstract:
        if options.pred_file == "" or options.actual_file == "":
            print("Missing input files - pred_file and actua_file!!!")
        else:
            utils.generate_abstract_text(options.pred_file, options.actual_file, options.show_graphs, batch_size=5)