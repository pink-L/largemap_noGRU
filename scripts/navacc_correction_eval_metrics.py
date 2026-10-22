"""Read-only, active first-episode CBF candidate/execution magnitude statistics."""
import torch
class CorrectionMetrics:
 def __init__(self,n,device):
  self.n=n;self.steps=torch.zeros(n,dtype=torch.long,device=device);self.defined=torch.zeros_like(self.steps);self.exec_defined=torch.zeros_like(self.steps);self.interventions=torch.zeros_like(self.steps);self.fallback=torch.zeros_like(self.steps)
  self.l2=torch.zeros(n,dtype=torch.float64,device=device);self.l2sq=torch.zeros_like(self.l2);self.exec_l2=torch.zeros_like(self.l2);self.values=[];self.exec_values=[]
 def observe(self,nxt,active):
  n=self.n;nom=nxt['info','planar_policy_nominal_acceleration'].reshape(n,3);candidate=nxt['info','planar_candidate_acceleration'].reshape(n,3);cmd=nxt['info','planar_commanded_acceleration'].reshape(n,3)
  valid=active & nxt['info','planar_cbf_correction_defined'].reshape(n).bool() & torch.isfinite(nom).all(-1) & torch.isfinite(candidate).all(-1)
  ev=active & torch.isfinite(nom).all(-1) & torch.isfinite(cmd).all(-1)
  d=torch.linalg.vector_norm(candidate.double()-nom.double(),dim=-1);ed=torch.linalg.vector_norm(cmd.double()-nom.double(),dim=-1)
  self.steps+=active.long();self.defined+=valid.long();self.exec_defined+=ev.long();self.interventions+=(valid&(d>1e-6)).long();self.fallback+=(active & nxt['info','planar_cbf_fallback'].reshape(n).bool()).long()
  self.l2+=torch.where(valid,d,0).double();self.l2sq+=torch.where(valid,d*d,0).double();self.exec_l2+=torch.where(ev,ed,0).double();self.values.append(d[valid].detach().clone());self.exec_values.append(ed[ev].detach().clone())
 def row(self,i):
  a=int(self.steps[i]);d=int(self.defined[i]);e=int(self.exec_defined[i]);return dict(cbf_observed_active_steps=a,cbf_candidate_defined_steps=d,cbf_candidate_defined_coverage=d/a if a else None,cbf_candidate_delta_l2_mean=float(self.l2[i]/d) if d else None,cbf_candidate_delta_l2_squared_mean=float(self.l2sq[i]/d) if d else None,cbf_execution_delta_l2_mean=float(self.exec_l2[i]/e) if e else None,cbf_execution_observed_steps=e,cbf_intervention_steps=int(self.interventions[i]),cbf_intervention_fraction_active=float(self.interventions[i]/a) if a else None,cbf_fallback_fraction_active=float(self.fallback[i]/a) if a else None)
 def summary(self):
  a=int(self.steps.sum());d=int(self.defined.sum());e=int(self.exec_defined.sum())
  def p95(values):
   x=torch.cat(values);return float(torch.quantile(x.double(),.95)) if len(x) else None
  return dict(schema='readonly_cbf_correction_metrics_v1',unit='m/s^2',candidate_definition='L2(planar_candidate_acceleration - planar_policy_nominal_acceleration), correction_defined AND finite',execution_definition='L2(planar_commanded_acceleration - planar_policy_nominal_acceleration), finite',active_steps=a,candidate_defined_steps=d,candidate_defined_coverage=d/a if a else None,candidate_delta_l2_mean=float(self.l2.sum()/d) if d else None,candidate_delta_l2_p95=p95(self.values),candidate_delta_l2_squared_mean=float(self.l2sq.sum()/d) if d else None,execution_observed_steps=e,execution_delta_l2_mean=float(self.exec_l2.sum()/e) if e else None,execution_delta_l2_p95=p95(self.exec_values),intervention_steps=int(self.interventions.sum()),intervention_threshold_m_s2=1e-6,intervention_fraction_active=float(self.interventions.sum()/a) if a else None,intervention_fraction_defined=float(self.interventions.sum()/d) if d else None,fallback_steps=int(self.fallback.sum()),fallback_fraction_active=float(self.fallback.sum()/a) if a else None,fallback_episode_count=int((self.fallback>0).sum()),episode_denominator=self.n,fallback_episode_incidence=float((self.fallback>0).float().mean()),no_actions_or_RNG_modified=True)
