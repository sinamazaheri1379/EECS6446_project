#!/bin/bash
#
# EECS6446 Elascale-Inspired Optimization Setup Script
# Based on: Khazaei et al. 2017 - "Elascale: Autoscaling and Monitoring as a Service"
#
# This script sets up the complete optimization environment for your Kubernetes cluster
#

set -e  # Exit on error

echo "============================================================"
echo "EECS6446 Elascale-Inspired HPA Optimization Setup"
echo "============================================================"

# Color codes for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ============================================================
# Step 1: Verify Prerequisites
# ============================================================
echo -e "\n${YELLOW}[Step 1/7] Verifying Prerequisites...${NC}"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}ERROR: kubectl not found. Please install kubectl first.${NC}"
    exit 1
fi

# Check if cluster is running
if ! kubectl get nodes &> /dev/null; then
    echo -e "${RED}ERROR: Kubernetes cluster not accessible. Please start your cluster first.${NC}"
    exit 1
fi

# Check if Online Boutique is deployed
if ! kubectl get deployment frontend -n default &> /dev/null; then
    echo -e "${RED}ERROR: Online Boutique not deployed. Please deploy it first using:${NC}"
    echo "kubectl apply -f files/online-boutique.yaml"
    exit 1
fi

echo -e "${GREEN}âœ“ Prerequisites verified${NC}"

# ============================================================
# Step 2: Install Python Dependencies
# ============================================================
echo -e "\n${YELLOW}[Step 2/7] Installing Python Dependencies...${NC}"

pip install --quiet pandas numpy matplotlib requests pyyaml kubernetes

echo -e "${GREEN}âœ“ Python dependencies installed${NC}"

# ============================================================
# Step 3: Create Results Directory
# ============================================================
echo -e "\n${YELLOW}[Step 3/7] Creating Directory Structure...${NC}"

mkdir -p /home/EECS6446_project/files/optimizations/results
mkdir -p /home/EECS6446_project/files/optimizations/monitoring
mkdir -p /home/EECS6446_project/files/optimizations/scaling
mkdir -p /home/EECS6446_project/files/optimizations/scripts
mkdir -p /home/EECS6446_project/files/optimizations/analysis

echo -e "${GREEN}âœ“ Directory structure created${NC}"

# ============================================================
# Step 4: Deploy Enhanced Prometheus Monitoring
# ============================================================
echo -e "\n${YELLOW}[Step 4/7] Configuring Enhanced Monitoring...${NC}"

# Create ServiceMonitor for better metrics collection
cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: ServiceAccount
metadata:
  name: prometheus
  namespace: default
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: prometheus
rules:
- apiGroups: [""]
  resources:
  - nodes
  - nodes/proxy
  - services
  - endpoints
  - pods
  verbs: ["get", "list", "watch"]
- apiGroups:
  - extensions
  resources:
  - ingresses
  verbs: ["get", "list", "watch"]
- nonResourceURLs: ["/metrics"]
  verbs: ["get"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: prometheus
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: prometheus
subjects:
- kind: ServiceAccount
  name: prometheus
  namespace: default
EOF

# Verify Prometheus is accessible
if kubectl get service prometheus-kube-prometheus-prometheus -n monitoring &> /dev/null; then
    echo -e "${GREEN}âœ“ Prometheus monitoring configured${NC}"
else
    echo -e "${YELLOW}âš  Prometheus service not found. Make sure it's deployed.${NC}"
fi

# ============================================================
# Step 5: Configure Resource Requests/Limits
# ============================================================
echo -e "\n${YELLOW}[Step 5/7] Configuring Resource Requests and Limits...${NC}"

# Based on Elascale paper - proper resource allocation is crucial
# for accurate autoscaling

services=("frontend" "cartservice" "checkoutservice" "productcatalogservice" "recommendationservice")

for service in "${services[@]}"; do
    echo "  Configuring $service..."
    
    kubectl set resources deployment/$service -n default \
        --requests=cpu=100m,memory=128Mi \
        --limits=cpu=500m,memory=512Mi \
        2>/dev/null || echo "    âš  Could not set resources for $service"
done

echo -e "${GREEN}âœ“ Resource requests/limits configured${NC}"

# ============================================================
# Step 6: Remove Existing HPAs (Clean Slate)
# ============================================================
echo -e "\n${YELLOW}[Step 6/7] Removing Existing HPAs...${NC}"

kubectl delete hpa --all -n default 2>/dev/null || true

echo -e "${GREEN}âœ“ Existing HPAs removed${NC}"

# ============================================================
# Step 7: Deploy Baseline HPA Configuration
# ============================================================
echo -e "\n${YELLOW}[Step 7/7] Deploying Baseline HPA Configuration...${NC}"

# Start with baseline for comparison
for service in "${services[@]}"; do
    kubectl autoscale deployment $service \
        --cpu-percent=70 \
        --min=1 \
        --max=10 \
        -n default \
        2>/dev/null || echo "  âš  Could not create HPA for $service"
done

echo -e "${GREEN}âœ“ Baseline HPA deployed${NC}"

# ============================================================
# Setup Complete
# ============================================================
echo -e "\n${GREEN}============================================================"
echo "Setup Complete! Your cluster is ready for optimization."
echo "============================================================${NC}"

echo -e "\nNext Steps:"
echo "1. Verify all services are running:"
echo "   ${YELLOW}kubectl get pods -n default${NC}"
echo ""
echo "2. Check HPA status:"
echo "   ${YELLOW}kubectl get hpa -n default${NC}"
echo ""
echo "3. Start port-forwarding for Prometheus (in another terminal):"
echo "   ${YELLOW}kubectl port-forward svc/prometheus-kube-prometheus-prometheus 9090:9090 -n monitoring${NC}"
echo ""
echo "4. Start port-forwarding for frontend (another terminal):"
echo "   ${YELLOW}kubectl port-forward svc/frontend 8080:8080 -n default${NC}"
echo ""
echo "5. Run the optimization experiment:"
echo "   ${YELLOW}python3 /home/claude/optimizations/scripts/elascale_mape_k_experiment.py${NC}"
echo ""
echo "6. Alternatively, apply Elascale HPA directly:"
echo "   ${YELLOW}kubectl apply -f /home/claude/optimizations/scaling/cartservice-elascale-hpa.yaml${NC}"
echo "   ${YELLOW}kubectl apply -f /home/claude/optimizations/scaling/services-elascale-hpa.yaml${NC}"

echo -e "\n${YELLOW}Note: Ensure Prometheus and frontend are accessible before running experiments${NC}"

# ============================================================
# Create Quick Reference Guide
# ============================================================
cat > /home/claude/optimizations/QUICK_REFERENCE.md <<'REFEOF'
# Elascale-Inspired Optimization - Quick Reference

## Key Commands

### View HPA Status
```bash
kubectl get hpa -n default
kubectl describe hpa frontend-elascale -n default
```

### Monitor Metrics
```bash
# Watch pod resource usage
kubectl top pods -n default

# Watch HPA scaling
watch kubectl get hpa -n default
```

### Apply Configurations
```bash
# Apply Elascale HPAs
kubectl apply -f /home/claude/optimizations/scaling/cartservice-elascale-hpa.yaml
kubectl apply -f /home/claude/optimizations/scaling/services-elascale-hpa.yaml
```

### Troubleshooting
```bash
# Check metrics-server
kubectl get deployment metrics-server -n kube-system

# Restart metrics-server if needed
kubectl rollout restart deployment metrics-server -n kube-system

# Check resource requests
kubectl describe deployment frontend | grep -A 3 "Requests"
```
REFEOF

echo -e "\n${GREEN}âœ“ Quick reference guide created at: /home/claude/optimizations/QUICK_REFERENCE.md${NC}"
