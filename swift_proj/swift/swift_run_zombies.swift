import io;
import sys;
import files;
import launch;
import python;
import stats;

string emews_root = getenv("EMEWS_PROJECT_ROOT");
string turbine_output = getenv("TURBINE_OUTPUT");

string model_main = "%s/../src/zombies/zombies.py" % emews_root;
string config_file = argv("config_file");
string param_file = argv("f"); // e.g. -f="model_params.txt";

//printf("Model Main = ");
//printf(model_main);

string json_param_code =
"""
import json
import os

instance = '%s'  # arg 1 is the instance folder
line = '%s'  # arg 2 is the UPF line as a JSON string

params_json = json.loads(line)

params_json['output.directory'] = instance

params_line = json.dumps(params_json)
    
""";

(int z[]) run_model()
{
    string param_lines[] = file_lines(input(param_file));
    
    //int z[];
    foreach s,i in param_lines
    {
        //printf(s);
        
        string instance = "%s/run_%d/" % (turbine_output, i);

        string code = json_param_code % (instance, s);
        string params_line = python_persist(code, "params_line");
        string params_line_wrapped = "'" + params_line + "'";
            
        // Swift special environment vars
        //string envs[] = ["swift_launcher=/usr/bin/srun"];
            
        // JCCM args
        string args[] = [model_main, config_file, params_line_wrapped];
        
        // TODO get par from command line
        
        // TODO need to create output folder using swift because model ranks will collide trying to write folder
        
        z[i] = @par=34 launch("python", args);
        //z[i] = @par=2 launch_envs("python", args, envs);
        
        printf("z[i] = %i", z[i]);

    }
    //v = propagate(sum(z));
}

main {
    z = run_model();
    printf("z = %i", sum_integer(z));
}
