export interface TaskContext {
  id: string;
  type: string;
  parameters: Record<string, any>;
  status: 'pending' | 'running' | 'completed' | 'failed';
  result?: any;
  createdAt: Date;
  updatedAt: Date;
}

export interface AgentConfig {
  id: string;
  name: string;
  type: string;
  capabilities: string[];
}