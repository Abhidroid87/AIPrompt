import { TaskContext, AgentConfig } from '../../types';

export interface IAgent {
  config: AgentConfig;
  status: 'idle' | 'busy' | 'error';
  
  initialize(): Promise<void>;
  executeTask(task: TaskContext): Promise<TaskContext>;
  handleError(error: Error): Promise<void>;
  cleanup(): Promise<void>;
  getStatus(): Promise<string>;
}