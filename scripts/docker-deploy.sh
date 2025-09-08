#!/bin/bash
# ESG Analytics IBEX35 - Docker Deployment Script
# ==============================================
# Automated deployment script for production environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_NAME="esg-analytics-ibex35"
COMPOSE_FILE="docker-compose.yml"
ENV_FILE=".env"

echo -e "${BLUE}🚀 ESG Analytics IBEX35 - Docker Deployment${NC}"
echo "=============================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker first.${NC}"
    exit 1
fi

# Check if docker-compose is available
if ! command -v docker-compose &> /dev/null; then
    echo -e "${YELLOW}⚠️  docker-compose not found. Using 'docker compose' instead.${NC}"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Function to display help
show_help() {
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  build       Build Docker images"
    echo "  logs        Show service logs"
    echo "  status      Show service status"
    echo "  clean       Clean up containers and images"
    echo "  help        Show this help message"
}

# Function to start services
start_services() {
    echo -e "${BLUE}📦 Starting ESG Analytics services...${NC}"
    
    # Create necessary directories
    mkdir -p data/{raw,processed,cleaned} logs reports/figures
    
    # Start services
    $DOCKER_COMPOSE up -d
    
    echo -e "${GREEN}✅ Services started successfully!${NC}"
    echo ""
    echo "🌐 Available services:"
    echo "  • API v2 (FastAPI):     http://localhost:8001"
    echo "  • API Documentation:    http://localhost:8001/docs"
    echo "  • Streamlit Dashboard:  http://localhost:8501"
    echo ""
    echo "📊 Check service status: $0 status"
}

# Function to stop services
stop_services() {
    echo -e "${YELLOW}🛑 Stopping ESG Analytics services...${NC}"
    $DOCKER_COMPOSE down
    echo -e "${GREEN}✅ Services stopped successfully!${NC}"
}

# Function to restart services
restart_services() {
    echo -e "${BLUE}🔄 Restarting ESG Analytics services...${NC}"
    $DOCKER_COMPOSE restart
    echo -e "${GREEN}✅ Services restarted successfully!${NC}"
}

# Function to build images
build_images() {
    echo -e "${BLUE}🔨 Building Docker images...${NC}"
    $DOCKER_COMPOSE build --no-cache
    echo -e "${GREEN}✅ Images built successfully!${NC}"
}

# Function to show logs
show_logs() {
    echo -e "${BLUE}📋 Showing service logs...${NC}"
    $DOCKER_COMPOSE logs -f
}

# Function to show status
show_status() {
    echo -e "${BLUE}📊 Service Status:${NC}"
    echo "=================="
    $DOCKER_COMPOSE ps
    
    echo ""
    echo -e "${BLUE}🔍 Health Checks:${NC}"
    echo "=================="
    
    # Check API health
    if curl -s http://localhost:8001/health > /dev/null 2>&1; then
        echo -e "API v2:     ${GREEN}✅ Healthy${NC}"
    else
        echo -e "API v2:     ${RED}❌ Unhealthy${NC}"
    fi
    
    # Check Dashboard
    if curl -s http://localhost:8501 > /dev/null 2>&1; then
        echo -e "Dashboard:  ${GREEN}✅ Healthy${NC}"
    else
        echo -e "Dashboard:  ${RED}❌ Unhealthy${NC}"
    fi
}

# Function to clean up
clean_up() {
    echo -e "${YELLOW}🧹 Cleaning up Docker resources...${NC}"
    
    # Stop and remove containers
    $DOCKER_COMPOSE down --remove-orphans
    
    # Remove unused images
    docker image prune -f
    
    # Remove unused volumes (be careful with this)
    read -p "Remove unused volumes? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker volume prune -f
    fi
    
    echo -e "${GREEN}✅ Cleanup completed!${NC}"
}

# Main script logic
case "${1:-help}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    build)
        build_images
        ;;
    logs)
        show_logs
        ;;
    status)
        show_status
        ;;
    clean)
        clean_up
        ;;
    help|*)
        show_help
        ;;
esac
